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
    stars_to_use = user_config['stars_to_use']

    # where we'll save our stamps
    regions_file = user_config['regions_path']

    # query frames
    frames = get_pandas(columns=['id', 'image_relpath', 'exptime', 'mjd', 'seeing_pixels', 'pixel_scale'],
                        conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    # for each frame, check if the PSF was already built -- else, go for it.
    for i, frame in frames.iterrows():
        stars = select_stars_for_a_frame(frame['id'], stars_to_use)
        if len(stars) == 0:
            # will deal with this later.
            raise
        psf_ref = 'psf_' + ''.join(sorted(stars['name']))

        # check so we don't redo for nothing
        if check_psf_exists(frame['id'], psf_ref):
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

        result = build_psf(datas, noisemaps, subsampling_factor=user_config['subsampling_factor'],
                           n_iter_analytic=user_config['n_iter_analytic'],
                           n_iter_adabelief=user_config['n_iter_pixels'],
                           masks=cosmics_masks)
        psf_plots_dir = user_config['plots_dir'] / psf_ref
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
            
        query = "INSERT INTO PSFs (frame_id, chi2, psf_ref) VALUES (?, ?, ?)"
        params = (frame['id'], float(result['chi2']), psf_ref)
        execute_sqlite_query(query, params, is_select=False)


"""

        imgname = image['imgname']
        t0 = time()

        # load stamps and noise maps for this image
        data, noisemap = getData(imgname)
        # open the file in which we'll store the result
        with h5py.File(psfsfile, 'r+') as f:
            # check if we need to build again
            if not redo and (imgname + '_residuals') in f.keys():
                print(imgname, 'already done and redo is False.')
                continue

            # call the routine defined above
            kwargs_final, narrowpsf, numpsf, moffat, fullmodel, residuals, extra_fields = buildPSF(data,
                                                                                                   noisemap)
            # time for storage. If key already exists, gotta delete it since
            # h5py does not like overwriting
            for to_store, name in zip([data, noisemap, narrowpsf, numpsf, moffat, fullmodel, residuals],
                                      ['data', 'noisemap', 'narrow', 'num', 'moffat', 'model', 'residuals']):
                key = f"{imgname}_{name}"
                if key in f.keys():
                    del f[key]
                f[key] = to_store

        # write plots
        if dopsfplots:
            try:
                # try because the analytical methods don't have a 'loss_history'
                # field.
                fig = plt.figure(figsize=(2.56, 2.56))
                plt.plot(extra_fields['loss_history'])
                plt.title('loss history')
                plt.tight_layout()
                with io.BytesIO() as buff:
                    # write the plot to a buffer, read it with numpy
                    fig.savefig(buff, format='raw')
                    buff.seek(0)
                    plotimg = np.frombuffer(buff.getvalue(), dtype=np.uint8)
                    w, h = fig.canvas.get_width_height()
                    # white <-> black:
                    lossim = 255 - plotimg.reshape((int(h), int(w), -1))[:, :, 0].T[:, ::-1]
                plt.close()
            except:
                print('no loss history in extra_fields')
                lossim = np.zeros((256, 256))
            plot_psf(image, noisemap, lossim)
        # time of iteration
        times.append(time() - t0)
"""
