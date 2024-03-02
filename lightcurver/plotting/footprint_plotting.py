import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from astropy.wcs import WCS


def plot_footprints(footprint_arrays, common_footprint, largest_footprint, save_path=None):
    plt.figure(figsize=(10, 8))
    if not common_footprint.is_empty:
        plt.fill(*largest_footprint.exterior.xy, alpha=0.2, color='purple', label='Largest Footprint')
    # Plot each footprint array
    for footprint_array in footprint_arrays:
        polygon = Polygon(footprint_array)
        x, y = polygon.exterior.xy
        plt.plot(x, y, color='red')
    if not common_footprint.is_empty:
        plt.fill(*common_footprint.exterior.xy, alpha=0.5, color='blue', label='Common Footprint')
    plt.legend()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)
