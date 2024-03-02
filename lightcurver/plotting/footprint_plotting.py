import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from astropy.wcs import WCS


def plot_footprints(wcs_footprints, common_footprint, save_path=None):
    # Plot and save the footprints
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection=WCS(wcs_footprints[0].to_header()))
    for footprint in wcs_footprints:
        plt.plot(*Polygon(footprint).exterior.xy, color='red')
    if not common_footprint.is_empty:
        plt.fill(*common_footprint.exterior.xy, alpha=0.5, color='blue', label='Common Footprint')
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)
