import numpy as np
import pandas as pd
from scipy.stats import sigmaclip
from copy import deepcopy
import warnings


def group_observations(df, threshold=0.8):
    """
    Groups observations in a DataFrame based on a time threshold.

    Parameters:
    - df: pandas DataFrame containing the observations.
        * Must include a 'mjd' column for Modified Julian Days (floats).
        * Must include pairs of 'fluxes_{ps}' and 'd_fluxes_{ps}' columns for each observed point source,
          where 'ps' is a unique identifier each point source.
        * May include other optional columns like 'seeing', for which the average will be computed in each bin.
    - threshold: float, the time threshold in days to define observation groups. Defaults to 0.8.

    Returns:
    - A new DataFrame containing the combined observations (weighted means with 2-sigmas rejection)
    """

    grouped_results = []
    df_sorted = df.sort_values(by='mjd')
    start_idx = 0

    point_sources = {col.split('_')[0] for col in df.columns if col.endswith('_flux') and not col.endswith('_d_flux')}


    for i in range(1, len(df_sorted)):
        if df_sorted.iloc[i]['mjd'] - df_sorted.iloc[i - 1]['mjd'] > threshold or i == len(df_sorted) - 1:
            end_idx = i if df_sorted.iloc[i]['mjd'] - df_sorted.iloc[i - 1]['mjd'] > threshold else i + 1
            df_group = df_sorted.iloc[start_idx:end_idx]
            avg_mjd = df_group['mjd'].mean()
            scatter_mjd = df_group['mjd'].std()

            this_epoch_group = {"mjd": avg_mjd, "scatter_mjd": scatter_mjd}
            flux_columns = [f'{ps}_flux' for ps in point_sources] + [f'{ps}_d_flux' for ps in point_sources]

            optional_averages = {col: df_group[col].mean() for col in df_group.columns if col not in (['mjd'] + flux_columns )}
            this_epoch_group.update(optional_averages)

            for ps in sorted(point_sources):
                col = f'{ps}_flux'
                d_col = f'{ps}_d_flux'
                curve_data = df_group[col].to_numpy()
                curve_variances = df_group[d_col].to_numpy() ** 2

                filtered_data, low_limit, high_limit = sigmaclip(curve_data, low=2, high=2)
                filtered_indices = np.logical_and(curve_data >= low_limit, curve_data <= high_limit)

                filtered_variances = curve_variances[filtered_indices]
                weights = 1. / filtered_variances
                weighted_mean = np.average(filtered_data, weights=weights)
                weighted_variance = np.average((filtered_data - weighted_mean) ** 2, weights=weights)
                weighted_std_deviation = np.sqrt(weighted_variance)

                this_epoch_group[f'{ps}_flux'] = weighted_mean
                this_epoch_group[f'{ps}_d_flux'] = np.sqrt(1./np.sum(weights))
                this_epoch_group[f'{ps}_scatter_flux'] = weighted_std_deviation
                this_epoch_group[f'{ps}_count_flux'] = len(weights)

            start_idx = end_idx
            grouped_results.append(this_epoch_group)

    return pd.DataFrame(grouped_results)


def convert_flux_to_magnitude(df):
    """
    Converts fluxes and flux errors/scatters (see params for pattern) in a DataFrame to magnitudes,
    assuming a 'zeropoint' column (zero if absent).
    This is very tailored to the output of the `group_observations` function.

    Parameters:
    - df: pandas DataFrame containing fluxes and flux errors.
          The DataFrame should contain columns named "{ps}_flux", "{ps}_d_flux", and "{ps}_flux_scatter"
          for each source {ps}, and optionally a "zeropoint" column.

    Returns:
    - A new pandas DataFrame with magnitudes and magnitude errors/scatters.
    """
    df = deepcopy(df)
    if 'zeropoint' not in df.columns:
        warnings.warn('Zeropoint column missing. Using a zeropoint of 0.', RuntimeWarning)
        df['zeropoint'] = 0
    aux_columns = [c for c in df.columns if '_scatter' in c or '_d_flux' in c or '_count' in c]
    flux_columns = [c for c in df.columns if '_flux' in c and c not in aux_columns]
    scatter_error_columns = [col for col in df.columns if '_scatter' in col]  # can be absent

    for flux_col in flux_columns:
        ps = flux_col.split('_')[0]
        error_col = f'{ps}_d_flux'

        df[f'{ps}_mag'] = -2.5 * np.log10(df[flux_col]) + df['zeropoint']

        df[f'{ps}_d_mag'] = 2.5 / np.log(10) * (df[error_col] / df[flux_col]).abs()

        scatter_col = f'{ps}_flux_scatter'
        if scatter_col in scatter_error_columns:
            df[f'{ps}_mag_scatter'] = 2.5 / np.log(10) * (df[scatter_col] / df[flux_col]).abs()

    return df
