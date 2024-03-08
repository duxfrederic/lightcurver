from scipy.stats import median_abs_deviation
import sqlite3

from ..structure.database import execute_sqlite_query, get_pandas
from ..structure.user_config import get_user_config
from ..utilities.footprint import get_combined_footprint_hash
from ..utilities.chi2_selector import get_chi2_bounds
from ..plotting.normalization_plotting import plot_normalized_star_curves


def get_fluxes(combined_footprint_hash, photometry_chi2_min, photometry_chi2_max):
    """
    Retrieves all the available star fluxes in all frames.
    If a given frame does not have a flux for this star, a NaN placeholder will be used.
    We filter by the chi2 of the fit: fluxes with an out of bounds chi2 will be replaced by a NaN
    value as well.

    :param combined_footprint_hash: int, the hash of the footprint we are working with.
    :param photometry_chi2_min: minimum acceptable chi2 value for the fit of the photometry of the star in this frame
    :param photometry_chi2_max: ditto but max
    :return: A list of frames that meet the criteria.
    """
    query = """
    SELECT s.name,
           f.id AS frame_id, 
           f.mjd,
           sff.star_gaia_id, 
           sff.combined_footprint_hash,
           IFNULL(sff.flux, 'NaN') AS flux,
           IFNULL(sff.flux_uncertainty, 'NaN') AS d_flux
    FROM 
       frames f
    JOIN star_flux_in_frame sff ON f.id = sff.frame_id 
    JOIN stars s ON sff.star_gaia_id = s.gaia_id AND sff.combined_footprint_hash = s.combined_footprint_hash
    WHERE 
        sff.combined_footprint_hash = ?
    AND 
        sff.chi2 BETWEEN ? AND ?
    ORDER BY 
       s.name, f.id"""
    params = (combined_footprint_hash, photometry_chi2_min, photometry_chi2_max)

    return execute_sqlite_query(query, params, is_select=True, use_pandas=True)


def update_normalization_coefficients(norm_data):
    db_path = get_user_config()['database_path']
    with sqlite3.connect(db_path, timeout=15.0) as conn:
        cursor = conn.cursor()

        # insert query with ON CONFLICT clause for bulk update
        # this handles the case where we try to insert normalization coefficients for a given frame again
        # on conflict, we update the existing record with the new coefficient values
        insert_query = """
        INSERT INTO normalization_coefficients (frame_id, combined_footprint_hash, coefficient, coefficient_uncertainty)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(combined_footprint_hash, frame_id) DO UPDATE SET
        coefficient=excluded.coefficient, coefficient_uncertainty=excluded.coefficient_uncertainty
        """

        cursor.executemany(insert_query, norm_data)
        conn.commit()


def calculate_coefficient():
    """
    this is a routine called by the workflow manager. It interfaces with the user config and the database
    to calculate a representative norm of each frame, which will be used as a normalization coefficient later.

    Returns:
        nothing

    """
    user_config = get_user_config()

    # query initial frames, so we can calculate the footprint at hand
    # TODO for all these frames_ini requests, if by the end of building this pipeline I do not
    # TODO see a variation in the pattern, just put them in the get_combined_footprint_hash function
    # TODO so it is called only when necessary.
    frames_ini = get_pandas(columns=['id', 'image_relpath', 'exptime', 'mjd', 'seeing_pixels', 'pixel_scale'],
                            conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_ini['id'].to_list())

    fluxes_fit_chi2_min, fluxes_fit_chi2_max = get_chi2_bounds(psf_or_fluxes='fluxes')
    df = get_fluxes(combined_footprint_hash=combined_footprint_hash,
                    photometry_chi2_min=fluxes_fit_chi2_min,
                    photometry_chi2_max=fluxes_fit_chi2_max)

    # get a reference flux by star as the mean of all fluxes for this star
    reference_flux = df.groupby('star_gaia_id')['flux'].mean().reset_index()
    reference_flux.rename(columns={'flux': 'reference_flux'}, inplace=True)

    # merge the reference flux with the original dataframe
    df_merged = df.merge(reference_flux, on='star_gaia_id')
    df_merged['relative_flux'] = df_merged['flux'] / df_merged['reference_flux']
    df_merged['relative_d_flux'] = df_merged['d_flux'] / df_merged['reference_flux']
    medians = df_merged.groupby('frame_id')['relative_flux'].median()
    mad = df_merged.groupby('frame_id')['relative_flux'].apply(median_abs_deviation)
    df_merged = df_merged.merge(medians.rename('median'), on='frame_id')
    df_merged = df_merged.merge(mad.rename('mad'), on='frame_id')
    n_star_per_frame = df_merged.groupby('frame_id')['relative_flux'].apply(len)
    df_merged = df_merged.merge(n_star_per_frame.rename('star_count_before_filtering'), on='frame_id')

    df_merged['z_score'] = abs((df_merged['relative_flux'] - df_merged['median']) / df_merged['mad'])

    df_filtered = df_merged[df_merged['z_score'] <= 3]
    n_star_per_frame = df_filtered.groupby('frame_id')['relative_flux'].apply(len)
    df_filtered = df_filtered.merge(n_star_per_frame.rename('star_count_after_filtering'), on='frame_id')
    df_filtered['weights'] = 1 / (df_filtered['relative_d_flux'])
    scatter = df_filtered.groupby('frame_id')['relative_flux'].std()
    df_filtered = df_filtered.merge(scatter.rename('scatter'), on='frame_id')

    weighted_avg = df_filtered.groupby('frame_id').apply(
        lambda x: (x['relative_flux'] * x['weights']).sum() / x['weights'].sum())
    df_filtered = df_filtered.merge(weighted_avg.rename('combined_rel_flux'), on='frame_id')
    weighted_error = df_filtered.groupby('frame_id')['relative_d_flux'].mean()
    df_filtered = df_filtered.merge(weighted_error.rename('combined_uncertainties'), on='frame_id')
    df_filtered['combined_uncertainties'] = df_filtered['combined_uncertainties'] + df_filtered['scatter']

    frame_normalization = df_filtered.groupby('frame_id').agg({
        'combined_rel_flux': 'mean',
        'combined_uncertainties': 'mean'
    }).reset_index()

    # rename for clarity
    frame_normalization.rename(columns={'combined_rel_flux': 'normalization',
                                        'combined_uncertainties': 'normalization_error'},
                               inplace=True)

    # ok, prepare the insert into the db
    norm_data = []
    for _, frame_norm in frame_normalization.iterrows():
        frame_id = frame_norm['frame_id']
        norm = float(frame_norm['normalization'])
        norm_err = float(frame_norm['normalization_error'])
        norm_data.append((frame_id, combined_footprint_hash, norm, norm_err))

    update_normalization_coefficients(norm_data)

    # ok, query it again in the plot function.
    plot_norm_dir = user_config['plots_dir'] / 'normalization' / str(combined_footprint_hash)
    plot_norm_dir.mkdir(exist_ok=True, parents=True)

    plot_file = plot_norm_dir / "normalization_fluxes_plot.pdf"
    plot_normalized_star_curves(combined_footprint_hash=combined_footprint_hash, save_path=plot_file)

