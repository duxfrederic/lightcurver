import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.visualization import simple_norm

cosmouline_df = pd.read_csv('J0030_VST_AD_cosmouline.csv')

lightcurver_df = pd.read_csv('J0030_VST_AD_lightcurver.csv')

plt.figure(figsize=(6, 4))

plt.errorbar(cosmouline_df['mhjd'], cosmouline_df['mag'], yerr=cosmouline_df['magerr'],
             fmt='o', color='orange', alpha=0.5, zorder=1, label='Cosmouline (legacy pipeline)',
             ecolor='gray')

plt.errorbar(lightcurver_df['mhjd'], lightcurver_df['mag'], yerr=lightcurver_df['magerr'],
             fmt='o', color='blue', alpha=0.5, zorder=2, label='LightCurver', ecolor='lightblue')

plt.gca().invert_yaxis()
plt.xlabel('Modified Julian Days')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim((19.44, 19.251))

def add_image_in_plot(image, ax, zoom, position, norm, x_offset, text):
    zoom_width = zoom_height = zoom * ax.get_figure().get_figwidth() / 100

    bbox = ax.get_position()
    x = bbox.x0 + x_offset
    y = bbox.y0 + 0.035
    width = zoom_width / ax.get_figure().get_figwidth()
    height = zoom_height / ax.get_figure().get_figheight()

    axins = inset_axes(ax, width=str(zoom) + '%', height=str(zoom) + '%', loc=position,
                       bbox_to_anchor=(x, y, width, height),
                       bbox_transform=ax.get_figure().transFigure, borderpad=0)
    axins.imshow(image, origin='lower', norm=norm, cmap='afmhot')
    axins.axis('off')
    axins.text(0.05, 0.99, text, transform=axins.transAxes, ha="left", va="top", fontsize=6, color="white")


hst_image = np.load('PSJ0030-1525_hst_image.npy')
raw_image = np.load('PSJ0030-1525_raw_image.npy')
deconv_image = np.load('PSJ0030-1525_deconv_image.npy')

ax = plt.gca()

initial_offset = 0.03
offset_increment = 0.145
zoom = 39

norm = simple_norm(raw_image)
add_image_in_plot(raw_image, ax, zoom=zoom, position='lower left', x_offset=initial_offset,
                  norm=norm, text='Single data cutout')
norm = simple_norm(deconv_image, stretch='asinh', asinh_a=1e-2, percent=99.8)
add_image_in_plot(deconv_image, ax, zoom=zoom, position='lower left', x_offset=initial_offset + offset_increment,
                  norm=norm, text='Joint deconvolution\nproduct')
norm = simple_norm(hst_image, stretch='asinh', asinh_a=1.5e-3, min_percent=3)
add_image_in_plot(hst_image, ax, zoom=zoom, position='lower left', x_offset=initial_offset + 2*offset_increment,
                  norm=norm, text='HST image')

plt.savefig('comparison_with_legacy_pipeline.jpg', bbox_inches='tight', pad_inches=0, dpi=400)

