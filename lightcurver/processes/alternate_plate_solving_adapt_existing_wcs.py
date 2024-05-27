import numpy as np
from astropy.io import fits
import astroalign as aa
from astropy.wcs import WCS
from astropy.table import Table
import logging

from ..structure.user_config import get_user_config
from ..structure.database import execute_sqlite_query
from .plate_solving import select_frames_needing_plate_solving, post_plate_solve_steps


def adapt_wcs(reference_wcs, reference_sources, target_sources):

    # first, extract the cd matrix from the reference wcs object
    if reference_wcs.wcs.has_cd():
        cdmatrix = reference_wcs.wcs.cd.copy()
    elif reference_wcs.wcs.has_pc():
        cdmatrix = reference_wcs.wcs.pc.copy()
        cdmatrix *= reference_wcs.wcs.cdelt
    else:
        raise AttributeError("No celestial WCS found in the provided WCS object.")

    transform, (match1, match2) = aa.find_transform(reference_sources, target_sources)

    # ok, now we transform the WCS.
    # if the pixels are transformed actively, the coordinates must be
    # transformed the other way to compensate:
    similarity = transform.params
    scaled_rotation = similarity[:2, :2]
    translation = similarity[:2, 2]

    # copy the ref wcs and inverse transform it:
    wcs_new = reference_wcs.deepcopy()
    # update the ref pixel
    refpixel = reference_wcs.wcs.crpix
    refpixel = np.dot(scaled_rotation, refpixel) + translation
    wcs_new.wcs.crpix = refpixel
    # rotation and scaling of the cd matrix.
    wcs_new.wcs.cd = np.dot(scaled_rotation, cdmatrix)

    return wcs_new, (match1, match2)


def alternate_plate_solve_adapt_ref():
    user_config = get_user_config()
    workdir = user_config['workdir']
    logger = logging.getLogger("lightcurver.alternate_plate_solving_adapt_existing_wcs")

    # select the frame to use as reference
    reference_frame_for_wcs = user_config['reference_frame_for_wcs']
    if reference_frame_for_wcs is not None:
        query = "select image_relpath, sources_relpath, id from frames where id = ?"
        frame_path, sources_path, ref_id = execute_sqlite_query(query, params=(reference_frame_for_wcs,),
                                                                is_select=True, use_pandas=False)[0]
    else:
        query = "select image_relpath, sources_relpath, id  from frames where plate_solved = 1 limit 1"
        frame_path, sources_path, ref_id = execute_sqlite_query(query, is_select=True, use_pandas=False)[0]

    # just pointing the relative paths to the correct absolute ones:
    frame_path = workdir / frame_path
    sources_path = workdir / sources_path

    logger.info(f'Attempting to align the WCS of frame {frame_path} onto more images.')
    header = fits.getheader(frame_path)
    reference_sources = Table(fits.getdata(sources_path))
    # unpack to match the needed format
    reference_sources = [(row['x'], row['y']) for row in reference_sources]
    logger.info(f'The reference frame {frame_path} has {len(reference_sources)} sources.')
    reference_wcs = WCS(header)
    if not reference_wcs.is_celestial:
        message = f'The WCS of the frame {frame_path} is not celestial. Aborting'
        logger.info(message)
        raise RuntimeError(message)

    # select the frames that need plate solving.
    frames = select_frames_needing_plate_solving(user_config=user_config, logger=logger)

    for ii, frame in frames.iterrows():

        target_sources = Table(fits.getdata(workdir / frame['sources_relpath']))
        # same as above:
        target_sources = [(row['x'], row['y']) for row in target_sources]

        try:
            wcs_new, (match1, match2) = adapt_wcs(reference_wcs=reference_wcs, reference_sources=reference_sources,
                                                  target_sources=target_sources)
            success = True
        except aa.MaxIterError:
            logger.log(f"Could not align frame {frame['id']}: max iterations reached before solution.")
            success = False
        except Exception as e:
            logger.log(f"I frame {frame['id']}: error, {e}")
            success = False

        if success:
            logger.info(f"Adapted WCS of frame {frame['id']}")
            with fits.open(workdir / frame['image_relpath'], mode="update") as hdul:
                hdul[0].header.update(wcs_new.to_header())
                hdul.flush()
            # post plate solve steps
            post_plate_solve_steps(frame_path=workdir / frame['image_relpath'],
                                   user_config=user_config, frame_id=frame['id'])

        # at the end, set the image to plate solved in db, and flag it as having had a plate solve attempt.
        execute_sqlite_query(query="UPDATE frames SET plate_solved = ?, attempted_plate_solve = 1 WHERE id = ?",
                             params=(1 if success else 0, frame['id']), is_select=False)



