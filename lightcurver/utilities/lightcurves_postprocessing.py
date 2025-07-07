import numpy as np
import pandas as pd
from scipy.stats import sigmaclip
from copy import deepcopy
import warnings


def group_observations(df, threshold=0.8):
    grouped_results = []
    df_sorted = df.sort_values(by='mjd')
    point_sources = {col.split('_')[0] for col in df.columns if col.endswith('_flux') and not col.endswith('_d_flux')}

    def process_group(df_group):
        avg_mjd = df_group['mjd'].mean()
        scatter_mjd = df_group['mjd'].std()
        # eh, make it 0 if only one element in group
        scatter_mjd = 0. if np.isnan(scatter_mjd) else scatter_mjd
        this_epoch_group = {"mjd": avg_mjd, "scatter_mjd": scatter_mjd}
        flux_columns = [f'{ps}_flux' for ps in point_sources] + [f'{ps}_d_flux' for ps in point_sources]
        optional_averages = {col: df_group[col].mean() for col in df_group.columns if col not in (['mjd'] + flux_columns)}
        this_epoch_group.update(optional_averages)
        for ps in sorted(point_sources):
            col = f'{ps}_flux'
            d_col = f'{ps}_d_flux'
            curve_data = df_group[col].to_numpy()
            curve_variances = df_group[d_col].to_numpy()**2
            filtered_data, low_limit, high_limit = sigmaclip(curve_data, low=2, high=2)
            filtered_indices = np.logical_and(curve_data >= low_limit, curve_data <= high_limit)
            filtered_variances = curve_variances[filtered_indices]
            if len(filtered_variances) > 0 and np.all(filtered_variances > 0):
                weights = 1. / filtered_variances
                weighted_mean = np.average(filtered_data, weights=weights)
                weighted_variance = np.average((filtered_data - weighted_mean)**2, weights=weights)
                weighted_std_deviation = np.sqrt(weighted_variance)
                d_flux = np.sqrt(1. / np.sum(weights))
                count_flux = len(filtered_variances)
            else:
                weighted_mean = float('nan')
                weighted_std_deviation = float('nan')
                d_flux = float('inf')
                count_flux = 0
            this_epoch_group[f'{ps}_flux'] = weighted_mean
            this_epoch_group[f'{ps}_d_flux'] = d_flux
            this_epoch_group[f'{ps}_scatter_flux'] = weighted_std_deviation
            this_epoch_group[f'{ps}_count_flux'] = count_flux
        return this_epoch_group

    start_idx = 0
    for i in range(1, len(df_sorted)):
        if df_sorted.iloc[i]['mjd'] - df_sorted.iloc[i - 1]['mjd'] > threshold:
            df_group = df_sorted.iloc[start_idx:i]
            grouped_results.append(process_group(df_group))
            start_idx = i
    if start_idx < len(df_sorted):
        df_group = df_sorted.iloc[start_idx:]
        grouped_results.append(process_group(df_group))
    return pd.DataFrame(grouped_results)


def convert_flux_to_magnitude(df):
    """
    Converts fluxes and flux errors/scatters in a DataFrame to magnitudes,
    assuming a 'zeropoint' column (zero if absent).
    This is very tailored to the output of the group_observations function.

    Parameters:
    - df: pandas DataFrame containing fluxes and flux errors.
          The DataFrame should contain columns named "{ps}_flux", "{ps}_d_flux", and optionally "{ps}_scatter_flux"
          for each source {ps}, and optionally a "zeropoint" column.

    Returns:
    - A new pandas DataFrame with magnitudes and magnitude errors/scatters.
      For each source {ps}, the following columns are added:
        - "{ps}_mag": Nominal magnitude
        - "{ps}_d_mag_down": Lower magnitude uncertainty
        - "{ps}_d_mag_up": Upper magnitude uncertainty
        - "{ps}_scatter_mag_down": Lower scatter magnitude (if applicable)
        - "{ps}_scatter_mag_up": Upper scatter magnitude (if applicable)
    """
    df = deepcopy(df)

    # check zeropoint present
    if 'zeropoint' not in df.columns:
        warnings.warn('Zeropoint column missing. Using a zeropoint of 0.', RuntimeWarning)
        df['zeropoint_used_in_conversion'] = 0.
        df['zeropoint'] = 0.

    # group relevant columns
    aux_columns = [c for c in df.columns if '_scatter_flux' in c or '_d_flux' in c or '_count' in c]
    flux_columns = [c for c in df.columns if '_flux' in c and c not in aux_columns]

    zeropoint = df['zeropoint']

    # some utility function to compute magnitude uncertainties from flux uncertainties
    def compute_mags_asymmetric_errors(flux_values, flux_errors, zp, source, prefix):
        """
        Helper function to compute magnitudes and asymmetric errors.

        Parameters:
        - F: Flux values.
        - dF: Flux uncertainties.
        - zp: Zeropoint values.
        - source: name of the source at hand
        - prefix: either 'd' or 'scatter' depending on the uncertainty type

        Returns:
        - Tuple of Series: (mag, sigma_down, sigma_up)
        """
        # safety
        flux_values = np.array(flux_values)
        flux_errors = np.array(flux_errors)
        # zp never goes through the jax machinery, should be a numpy array.

        # nominal magnitude
        mag = -2.5 * np.log10(flux_values) + zp
        # upper and lower fluxes
        flux_up = flux_values + flux_errors
        flux_down = flux_values - flux_errors
        # start: NaNs
        mag_down = np.full_like(mag, np.nan, dtype=np.float64)
        mag_up = np.full_like(mag, np.nan, dtype=np.float64)
        # validity masks
        valid_plus = flux_up > 0
        valid_minus = flux_down > 0
        # calc magnitudes where valid
        mag_down[valid_plus] = -2.5 * np.log10(flux_up[valid_plus]) + zp[valid_plus]
        mag_up[valid_minus] = -2.5 * np.log10(flux_down[valid_minus]) + zp[valid_minus]
        # asymmetric errors
        sigma_down = mag - mag_down
        sigma_up = mag_up - mag
        # assign
        df[f'{source}_mag'] = mag
        df[f'{source}_{prefix}_mag_down'] = sigma_down
        df[f'{source}_{prefix}_mag_up'] = sigma_up
        # linearized mag uncertainty for comparison
        df[f'{source}_{prefix}_mag'] = 2.5 / np.log(10) * np.abs(flux_errors / flux_values)

    for error_type in ('d', 'scatter'):
        for flux_col in flux_columns:
            ps = flux_col.split('_')[0]
            error_col = f'{ps}_{error_type}_flux'
            if error_col in df.columns:
                compute_mags_asymmetric_errors(flux_values=df[flux_col],
                                               flux_errors=df[error_col],
                                               zp=zeropoint,
                                               source=ps,
                                               prefix=error_type)

    return df
