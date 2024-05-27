import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def plot_footprints(footprint_arrays, common_footprint=None, largest_footprint=None, save_path=None):
    """
    shows the polygons representing the footprints of frames on a plot.

    """
    fig = plt.figure(figsize=(10, 8))
    if (largest_footprint is not None) and (not largest_footprint.is_empty):
        plt.fill(*largest_footprint.exterior.xy, alpha=0.2, color='purple', label='Largest Footprint')
    # Plot each footprint array
    for footprint_array in footprint_arrays:
        polygon = Polygon(footprint_array)
        x, y = polygon.exterior.xy
        plt.plot(x, y, color='gray', lw=0.5)
    if (common_footprint is not None) and (not common_footprint.is_empty):
        plt.fill(*common_footprint.exterior.xy, alpha=0.5, color='blue', label='Common Footprint')
    plt.legend()
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)
        plt.close()
    else:
        return fig, plt.gca()


