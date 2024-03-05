import numpy as np
from pathlib import Path
import h5py
from starred.procedures.psf_routines import build_psf

from ..structure.database import select_stars_for_a_frame, execute_sqlite_query, get_pandas
from ..structure.user_config import get_user_config
from ..plotting.psf_plotting import plot_psf_diagnostic


def check_psf_exists(frame_id, psf_ref):
    query = "SELECT 1 FROM PSFs WHERE frame_id = ? AND psf_ref = ?"
    params = (frame_id, psf_ref)
    result = execute_sqlite_query(query, params)
    return len(result) > 0


def model_all_psfs():
    user_config = get_user_config()
    stars_to_use = user_config['stars_to_use_psf']

    # where we'll save our stamps
    regions_file = user_config['regions_path']

    # query frames
    frames = get_pandas(columns=['id', 'image_relpath', 'exptime', 'mjd', 'seeing_pixels', 'pixel_scale'],
                        conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    # for each frame, check if the PSF was already built -- else, go for it.
    for i, frame in frames.iterrows():
        if frame['id'] != 33:
            continue
        stars = select_stars_for_a_frame(frame['id'], stars_to_use)
        if len(stars) == 0:
            # will deal with this later.
            raise RuntimeError("No star in this frame!")
        psf_ref = 'psf_' + ''.join(sorted(stars['name']))

        # check so we don't redo for nothing
        if check_psf_exists(frame['id'], psf_ref) and not user_config['redo_psf']:
            continue

        # get the cutouts
        with h5py.File(regions_file, 'r') as f:
            data_group = f[f"{frame['image_relpath']}/data"]
            noisemap_group = f[f"{frame['image_relpath']}/noisemap"]
            mask_group = f[f"{frame['image_relpath']}/cosmicsmask"]
            datas = np.array([data_group[name][...] for name in sorted(stars['name'])])
            noisemaps = np.array([noisemap_group[name][...] for name in sorted(stars['name'])])
            cosmics_masks = np.array([mask_group[name][...] for name in sorted(stars['name'])]).astype(bool)
            # invert because the cosmics are marked as True, but we want the healthy pixels to be marked as True:
            cosmics_masks = ~cosmics_masks
        isnan = np.where(np.isnan(datas)*np.isnan(noisemaps))
        datas[isnan] = 0.
        noisemaps[isnan] = 1.0
        cosmics_masks[isnan] = False
        # we set the initial guess for the position of the star to the center (guess_method thing)
        # because we are confident that is where the star will be (plate solving + gaia proper motions)
        result = build_psf(datas, noisemaps, subsampling_factor=user_config['subsampling_factor'],
                           n_iter_analytic=user_config['n_iter_analytic'],
                           n_iter_adabelief=user_config['n_iter_pixels'],
                           masks=cosmics_masks,
                           guess_method_star_position='center')
        psf_plots_dir = user_config['plots_dir'] / 'PSFs'
        psf_plots_dir.mkdir(exist_ok=True)
        frame_name = Path(frame['image_relpath']).stem
        seeing = frame['seeing_pixels'] * frame['pixel_scale']
        plot_psf_diagnostic(datas=datas, noisemaps=noisemaps, residuals=result['residuals'],
                            full_psf=result['full_psf'],
                            loss_curve=result['adabelief_extra_fields']['loss_history'],
                            masks=cosmics_masks, names=sorted(stars['name']),
                            diagnostic_text=f"{frame_name}\nseeing: {seeing:.02f}",
                            save_path=psf_plots_dir / f"{frame_name}.jpg")

        # now we can do the bookkeeping stuff
        with h5py.File(regions_file, 'r+') as f:
            # if already there, delete it before replacing.
            frame_group = f[frame['image_relpath']]
            if psf_ref in frame_group.keys():
                del frame_group[psf_ref]
            psf_group = frame_group.create_group(psf_ref)
            psf_group['narrow_psf'] = np.array(result['narrow_psf'])
            psf_group['full_psf'] = np.array(result['full_psf'])

        # and update the database.
        delete_query = "DELETE FROM PSFs WHERE frame_id = ? AND psf_ref = ?"
        delete_params = (frame['id'], psf_ref)
        execute_sqlite_query(delete_query, delete_params, is_select=False)
        insert_query = "INSERT INTO PSFs (frame_id, chi2, psf_ref) VALUES (?, ?, ?)"
        insert_params = (frame['id'], float(result['chi2']), psf_ref)
        execute_sqlite_query(insert_query, insert_params, is_select=False)


