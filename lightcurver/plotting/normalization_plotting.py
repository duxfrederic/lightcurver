import matplotlib.pyplot as plt


def plot_normalized_star_curves(combined_footprint):
    # TODO
    # Merge the frame normalization data back into the merged dataset
    df_final = None#df_merged.merge(frame_normalization[['frame_id', 'normalization']], on='frame_id')

    # Calculate the normalized flux for each measurement
    df_final['normalized_flux'] = df_final['flux'] / df_final['normalization']

    selected_stars = df_final['star_gaia_id'].unique()[:5]

    plt.figure(figsize=(14, 10))

    for star_id in selected_stars:
        star_data = df_final[df_final['star_gaia_id'] == star_id]
        plt.errorbar(star_data['frame_id'], star_data['normalized_flux'],
                     yerr=star_data['d_flux'], fmt='.-', label=f'Star {star_id}')

    plt.xlabel('Frame ID')
    plt.ylabel('Normalized Flux')
    plt.legend()
    plt.grid(True)
    plt.show()

