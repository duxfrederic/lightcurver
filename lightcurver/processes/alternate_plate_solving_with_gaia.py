# this is for the rare cases where standard plate solving fails
# we use our gaia stars and detections to create a reasonable WCS for our frames.
# to be attempted if all else fails.
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.table import Table
from astropy.io import fits
import astroalign as aa
import sep

from ..structure.database import execute_sqlite_query, get_pandas
from ..utilities.gaia import query_gaia_stars
from ..plotting.sources_plotting import plot_coordinates_and_sources_on_image
from ..processes.plate_solving import post_plate_solve_steps


def create_initial_wcs(pixel_scale, image_shape, center_ra, center_dec, rotation_angle_deg):
    w = WCS(naxis=2)
    w.wcs.crpix = [(image_shape[1] - 1) / 2, (image_shape[0] - 1) / 2]  # center of the image
    w.wcs.crval = [center_ra, center_dec]  # ra, dec at the center
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    pixel_scale_deg = pixel_scale / 3600.0
    w.wcs.cd = np.array([[-pixel_scale_deg * np.cos(rotation_angle_rad), pixel_scale_deg * np.sin(rotation_angle_rad)],
                         [pixel_scale_deg * np.sin(rotation_angle_rad), pixel_scale_deg * np.cos(rotation_angle_rad)]])

    return w


def refine_wcs_with_astroalign(sources, gaia_star_coords, wcs):
    """
    takes in a 'guess' wcs object (made by a helper function in this file), crossmatches the resulting
    pixel positions of the gaia stars to the extracted sources, and if possible calculates a WCS.
    Args:
        sources: astropy Table of sources, with 'xcentroid' and 'ycentroid' columns
        gaia_star_coords: astropy SkyCoord containing the coordinates of our stars (please correct for proper motion)
        wcs: an initial guess wcs object

    Returns:

    """

    gaia_pix_coords = wcs.world_to_pixel(gaia_star_coords)
    gaia_pix_positions = np.vstack(gaia_pix_coords).T

    image_positions = np.column_stack((sources['x'], sources['y']))

    try:
        transf, (source_idx, gaia_idx) = aa.find_transform(image_positions, gaia_pix_positions)
    except Exception as e:
        print(f"Error finding transformation with astroalign: {e}")
        raise

    crpix = transf.inverse.params[:2, :2] @ wcs.wcs.crpix + transf.inverse.translation

    new_cd = transf.inverse.params[:2, :2] @ wcs.wcs.cd
    wcs_updated = wcs.deepcopy()
    wcs_updated.wcs.crpix = crpix
    wcs_updated.wcs.cd = new_cd

    return wcs_updated


def alternate_plate_solve(user_config):

    ra, dec = user_config['ROI_ra_deg'], user_config['ROI_dec_deg']
    center_radius = {'center': (ra, dec), 'radius':  user_config['alternate_plate_solve_gaia_radius']/3600.}
    gaia_stars = query_gaia_stars('circle', center_radius=center_radius)
    gaia_stars['pmra'][np.isnan(gaia_stars['pmra'])] = 0
    gaia_stars['pmdec'][np.isnan(gaia_stars['pmdec'])] = 0
    gaia_coords = SkyCoord(ra=gaia_stars['ra'],
                           dec=gaia_stars['dec'],
                           pm_ra_cosdec=gaia_stars['pmra'],
                           pm_dec=gaia_stars['pmdec'],
                           frame='icrs',
                           obstime=Time(gaia_stars['ref_epoch'].value, format='decimalyear'))
    pixel_scale = np.mean(user_config['plate_scale_interval'])

    frames_to_process = get_pandas(columns=['id', 'image_relpath', 'sources_relpath', 'mjd'],
                                   conditions=['plate_solved = 0', 'eliminated = 0'])
    plot_dir = user_config['plots_dir'] / 'gaia_plate_solve_diagnostic'
    plot_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in frames_to_process.iterrows():

        gaia_coords_moved = gaia_coords.apply_space_motion(new_obstime=Time(frame['mjd'], format='mjd'))
        frame_path = user_config['workdir'] / frame['image_relpath']
        frame_id = frame['id']
        image = fits.getdata(frame_path).astype(float)
        # sources_path = user_config['workdir'] / frame['sources_relpath']
        # sources = Table(fits.getdata(sources_path))
        bck = sep.Background(image.astype(float), bw=128, bh=128)
        sources = Table(sep.extract(image, thresh=3., err=bck.globalrms, minarea=15))
        initial_wcs = create_initial_wcs(pixel_scale=pixel_scale, center_ra=ra, center_dec=dec, rotation_angle_deg=0,
                                         image_shape=image.shape)
        try:
            new_wcs = refine_wcs_with_astroalign(sources, gaia_coords_moved, initial_wcs)
            success = True
        except Exception as e:
            print(f"Could not solve frame {frame_id}: {e}.")
            success = False
        if success:
            with fits.open(frame_path, mode="update") as hdul:
                hdul[0].header.update(new_wcs.to_header())
                hdul.flush()
            plot_path = plot_dir / f"{frame_path.stem}.jpg"
            plot_coordinates_and_sources_on_image(image, sources=sources,
                                                  gaia_coords=gaia_coords_moved, wcs=new_wcs, save_path=plot_path)
            post_plate_solve_steps(frame_path=frame_path, user_config=user_config, frame_id=frame_id)

        # at the end, set the image to plate solved in db
        execute_sqlite_query(query="UPDATE frames SET plate_solved = ? WHERE id = ?",
                             params=(1 if success else 0, frame_id), is_select=False)
