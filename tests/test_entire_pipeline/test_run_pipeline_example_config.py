import os
import tempfile
import shutil
import yaml
import sqlite3
from lightcurver.pipeline.workflow_manager import WorkflowManager
from lightcurver.pipeline.task_wrappers import source_extract_all_images


def database_checks(db_path):
    """
    Performs various checks on the database, including chi2 values and data integrity.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # chi2 checks
    cursor.execute("SELECT COUNT(*) FROM PSFs WHERE chi2 >= 2;")
    assert cursor.fetchone()[0] == 0, "There are PSF models with chi2 >= 2."
    cursor.execute("SELECT COUNT(*) FROM star_flux_in_frame WHERE chi2 >= 2;")
    assert cursor.fetchone()[0] == 0, "There are stars_in_flux_in_frame values with chi2 >= 2."

    # count check
    cursor.execute("SELECT COUNT(*) FROM PSFs;")
    n_psf = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM frames;")
    n_frames = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM normalization_coefficients;")
    n_coeffs = cursor.fetchone()[0]

    assert n_psf == n_frames == n_coeffs, f"not same number of coeffs, psfs, frames: {n_coeffs, n_psf, n_frames}"

    conn.close()


def test_run_workflow():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # paths relative to the repository root
    config_path = os.path.join(current_dir, '..', '..', 'lightcurver', 'pipeline',
                               'example_config_file', 'config.yaml')
    header_function_path = os.path.join(current_dir, '..', '..', 'docs',
                                        'example_header_parser_functions', 'parse_omegacam_header.py')
    data_path = os.path.join(current_dir, 'raw_frames')

    # temp dir setup
    temp_dir = tempfile.mkdtemp(prefix='lightcurver_test_')

    # modify the configuration: has to point to tempdir
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config['workdir'] = temp_dir
    config['raw_dirs'] = [data_path]
    config['already_plate_solved'] = 1
    config['ROI_disk_radius_arcseconds'] = 100
    config['stars_to_use_psf'] = config['stars_to_use_norm'] = 2
    config['stamp_size_stars'] = config['stamp_size_roi'] = 24
    config['multiprocessing_cpu_count'] = 2  # what GitHub gives us I think

    # save the modified configuration to a temporary file
    temp_config_path = os.path.join(temp_dir, 'config_temp.yaml')
    with open(temp_config_path, 'w') as file:
        yaml.safe_dump(config, file)

    # copy the header function
    header_parser_dir = os.path.join(temp_dir, 'header_parser')
    os.mkdir(header_parser_dir)
    shutil.copy2(header_function_path, os.path.join(header_parser_dir, 'parse_header.py'))

    os.environ['LIGHTCURVER_CONFIG'] = temp_config_path

    # first run the importation
    wf_manager = WorkflowManager()
    wf_manager.run(final_step='query_gaia_for_stars')

    # for this test, we'll also pretend that one of the images does not have an astrometric solution.
    # so, set the mode we want to test in the config:
    with open(temp_config_path, 'r') as file:
        config = yaml.safe_load(file)
    # this is the mode we are testing:
    config['plate_solving_strategy'] = 'adapt_wcs_from_reference'
    # because we do not want to depend on astrometry.net for this test.
    config['already_plate_solved'] = 0
    with open(temp_config_path, 'w') as file:
        yaml.safe_dump(config, file)
    # ready to run:
    wf_manager = WorkflowManager()
    wf_manager.run(final_step='plate_solving')
    # just before the plate solving, let us indicate that one of the images has an astrometric solution
    db_path = os.path.join(temp_dir, 'database.sqlite3')
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("update frames set plate_solved = 0, attempted_plate_solve = 0 where id = 1")
        cursor.execute("update frames set plate_solved = 1, attempted_plate_solve = 1 where id = 2")
    # and now run the whole pipeline.
    wf_manager.run()

    # some basic database checks: did the psf fits go well? same for photometry.
    # do we indeed have 2 psfs?
    # do we have 2 normalization coefficients?
    database_checks(db_path)

    # now, the user might want to redo the source extraction should the initial one not have
    # given the expected result. Test that it works without error here:
    source_extract_all_images()

    # now, redo everything but with the possibility of distorting the PSF. Namely, we start from the psf state by
    # removing all psfs from the database.
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM PSFs")

    # set distortion to true, and while we're at it a different calibration (using pan-starrs)
    with open(temp_config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['field_distortion'] = True
    config['photometric_band'] = 'r_panstarrs'
    # not redoing roi modelling to things speed up
    config['do_ROI_model'] = False
    with open(temp_config_path, 'w') as file:
        yaml.safe_dump(config, file)

    # and run again without error!
    wf_manager.run()




