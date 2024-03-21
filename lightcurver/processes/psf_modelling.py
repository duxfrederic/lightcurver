import numpy as np
from pathlib import Path
import h5py
import sep
from starred.procedures.psf_routines import build_psf

from ..structure.database import select_stars_for_a_frame, execute_sqlite_query, get_pandas
from ..structure.user_config import get_user_config
from ..plotting.psf_plotting import plot_psf_diagnostic
from ..utilities.footprint import get_combined_footprint_hash


def check_psf_exists(frame_id, psf_ref, combined_footprint_hash):
    query = "SELECT 1 FROM PSFs WHERE frame_id = ? AND psf_ref = ? and combined_footprint_hash = ?"
    params = (frame_id, psf_ref, combined_footprint_hash)
    result = execute_sqlite_query(query, params)
    return len(result) > 0


def mask_surrounding_stars(data, noisemap):
    """
    masks any object that is not the central one. (pixel ok = True, pixel masked = False)
    Args:
        data: 2d array
        noisemap:  2d array

    Returns:
        mask (2d array, one if good pixel, zero if masked)
    """
    objects, seg_map = sep.extract(data, thresh=3., err=noisemap, minarea=15, segmentation_map=True,
                                   deblend_cont=0.001)

    mask = np.ones_like(seg_map, dtype=bool)
    if len(objects) == 0:
        return mask
    center_y = (data.shape[0] - 1) / 2.0
    center_x = (data.shape[1] - 1) / 2.0
    distances = np.sqrt((objects['x'] - center_x) ** 2 + (objects['y'] - center_y) ** 2)

    central_star_index = np.argmin(distances)

    for i in range(0, len(objects)):
        if i != central_star_index:
            mask[seg_map == i + 1] = False

    return mask


def model_all_psfs():
    user_config = get_user_config()
    stars_to_use = user_config['stars_to_use_psf']

    # where we'll save our stamps
    regions_file = user_config['regions_path']

    # query frames
    frames = get_pandas(columns=['id', 'image_relpath', 'exptime', 'mjd', 'seeing_pixels', 'pixel_scale'],
                        conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames['id'].to_list())
    # for each frame, check if the PSF was already built -- else, go for it.
    from time import time

    for i, frame in frames.iterrows():
        t0 = time()
        stars = select_stars_for_a_frame(frame_id=frame['id'],
                                         combined_footprint_hash=combined_footprint_hash,
                                         stars_to_use=stars_to_use)
        stars.sort_values(by=['name'])
        if len(stars) == 0:
            # we simply do not build a PSF. the frame will not be considered in the joint queries later.
            continue
        psf_ref = 'psf_' + ''.join(sorted(stars['name']))

        # check so we don't redo for nothing
        if check_psf_exists(frame['id'], psf_ref, combined_footprint_hash) and not user_config['redo_psf']:
            continue

        # get the cutouts
        with h5py.File(regions_file, 'r') as f:
            data_group = f[f"{frame['image_relpath']}/data"]
            noisemap_group = f[f"{frame['image_relpath']}/noisemap"]
            mask_group = f[f"{frame['image_relpath']}/cosmicsmask"]
            datas = np.array([data_group[gaia_id][...] for gaia_id in list(stars['gaia_id'])])
            noisemaps = np.array([noisemap_group[name][...] for name in list(stars['gaia_id'])])
            cosmics_masks = np.array([mask_group[name][...] for name in list(stars['gaia_id'])]).astype(bool)
            # invert because the cosmics are marked as True, but we want the healthy pixels to be marked as True:
            cosmics_masks = ~cosmics_masks
        # now we'll prepare automatic masks (masking other objects in the field)
        automatic_masks = []
        for data, noisemap in zip(datas, noisemaps):
            mask = mask_surrounding_stars(data, noisemap)
            automatic_masks.append(mask)
        automatic_masks = np.array(automatic_masks)
        # now both cosmics_masks and automatic_masks have 0 for bad pixels, and 1 for good pixels.
        masks = cosmics_masks * automatic_masks
        isnan = np.where(np.isnan(datas)*np.isnan(noisemaps))
        # before eliminating entire slices, 'mask' the NaNs
        datas[isnan] = 0.
        noisemaps[isnan] = 1.0
        masks[isnan] = False
        # now eliminate slices, which changes the shapes.
        # we'll delete the slices with too many masked pixels (we mask anything with more than 40% masked)
        mask_threshold_fraction = 0.4
        total_elements_per_slice = datas.shape[1] * datas.shape[2]
        masked_counts_per_slice = np.sum(~masks, axis=(1, 2))
        slices_to_remove = masked_counts_per_slice > (mask_threshold_fraction * total_elements_per_slice)
        slices_to_keep = ~slices_to_remove
        datas = datas[slices_to_keep]
        noisemaps = noisemaps[slices_to_keep]
        masks = masks[slices_to_keep]
        names = list(stars['name'][slices_to_keep])
        if len(datas) == 0:
            # no psf model for this one.
            continue

        # we set the initial guess for the position of the star to the center (guess_method thing)
        # because we are confident that is where the star will be (plate solving + gaia proper motions)
        result = build_psf(datas, noisemaps, subsampling_factor=user_config['subsampling_factor'],
                           n_iter_analytic=user_config['psf_n_iter_analytic'],
                           n_iter_adabelief=user_config['psf_n_iter_pixels'],
                           masks=masks,
                           guess_method_star_position='center')
        psf_plots_dir = user_config['plots_dir'] / 'PSFs' / str(combined_footprint_hash)
        psf_plots_dir.mkdir(exist_ok=True, parents=True)
        frame_name = Path(frame['image_relpath']).stem
        seeing = frame['seeing_pixels'] * frame['pixel_scale']
        loss_history = result['adabelief_extra_fields']['loss_history']
        plot_psf_diagnostic(datas=datas, noisemaps=noisemaps, residuals=result['residuals'],
                            full_psf=result['full_psf'],
                            loss_curve=loss_history,
                            masks=masks, names=names,
                            diagnostic_text=f"{frame_name}\nseeing: {seeing:.02f}",
                            save_path=psf_plots_dir / f"{frame['id']}_{frame_name}.jpg")

        # now we can do the bookkeeping stuff
        with h5py.File(regions_file, 'r+') as f:
            # if already there, delete it before replacing.
            frame_group = f[frame['image_relpath']]
            if psf_ref in frame_group.keys():
                del frame_group[psf_ref]
            psf_group = frame_group.create_group(psf_ref)
            psf_group['narrow_psf'] = np.array(result['narrow_psf'])
            psf_group['full_psf'] = np.array(result['full_psf'])
            psf_group['subsampling_factor'] = np.array([user_config['subsampling_factor']])

        # and update the database.
        loss_index = int(0.9 * np.array(loss_history).size)
        initial_change = np.nanmax(loss_history[:loss_index]) - np.nanmin(loss_history[:loss_index])
        end_change = np.nanmax(loss_history[loss_index:]) - np.nanmin(loss_history[loss_index:])
        relative_loss_differential = float(end_change / initial_change)
        insert_params = (frame['id'], float(result['chi2']), relative_loss_differential, psf_ref,
                         combined_footprint_hash, user_config['subsampling_factor'])
        insert_query = "REPLACE INTO PSFs "
        insert_query += "(frame_id, chi2, relative_loss_differential, psf_ref, combined_footprint_hash, subsampling_factor) "
        insert_query += f"VALUES ({','.join(['?'] * len(insert_params))})"

        execute_sqlite_query(insert_query, insert_params, is_select=False)



