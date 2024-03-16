import matplotlib.pyplot as plt
import pandas as pd

cosmouline_df = pd.read_csv('J0659_VST_D_cosmouline.csv')
lightcurver_df = pd.read_csv('J0659_VST_D_lightcurver.csv')

plt.figure(figsize=(8, 6))

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

plt.savefig('comparison_with_legacy_pipeline.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
