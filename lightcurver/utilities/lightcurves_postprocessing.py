import numpy as np
import pandas as pd
from scipy.stats import sigmaclip


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

    point_sources = {col.split('_')[1] for col in df.columns if col.startswith('fluxes_')}

    for i in range(1, len(df_sorted)):
        if df_sorted.iloc[i]['mjd'] - df_sorted.iloc[i - 1]['mjd'] > threshold or i == len(df_sorted) - 1:
            end_idx = i if df_sorted.iloc[i]['mjd'] - df_sorted.iloc[i - 1]['mjd'] > threshold else i + 1
            df_group = df_sorted.iloc[start_idx:end_idx]
            avg_mjd = df_group['mjd'].mean()
            scatter_mjd = df_group['mjd'].std()

            this_epoch_group = {"average_mjd": avg_mjd, "scatter_mjd": scatter_mjd}

            for ps in point_sources:
                col = f'fluxes_{ps}'
                d_col = f'd_fluxes_{ps}'
                curve_data = df_group[col].to_numpy()
                curve_variances = df_group[d_col].to_numpy() ** 2

                filtered_data, low_limit, high_limit = sigmaclip(curve_data, low=2, high=2)
                filtered_indices = np.logical_and(curve_data >= low_limit, curve_data <= high_limit)

                filtered_variances = curve_variances[filtered_indices]
                weights = 1. / filtered_variances
                weighted_mean = np.average(filtered_data, weights=weights)
                weighted_variance = np.average((filtered_data - weighted_mean) ** 2, weights=weights)
                weighted_std_deviation = np.sqrt(weighted_variance)

                this_epoch_group[f'flux_{ps}'] = weighted_mean
                this_epoch_group[f'd_flux_{ps}'] = np.sqrt(1./np.sum(weights))
                this_epoch_group[f'scatter_flux_{ps}'] = weighted_std_deviation
                this_epoch_group[f'count_flux_{ps}'] = len(weights)

            start_idx = end_idx
            grouped_results.append(this_epoch_group)

    return pd.DataFrame(grouped_results)
