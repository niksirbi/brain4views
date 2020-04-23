import time
import numpy as np
from matplotlib import pyplot as plt
from nilearn.surface import vol_to_surf
from brain4views import plot_surf4

start = time.time()

lh_surf = './macaque_data/NMT13_lh.pial'
rh_surf = './macaque_data/NMT13_rh.pial'
stat = './macaque_data/example_stat_inNMT13.nii.gz'
lh_over = vol_to_surf(stat, lh_surf, radius=1, kind='ball')
rh_over = vol_to_surf(stat, rh_surf, radius=1, kind='ball')

# Plot unthresholded correlation map
plot_surf4([lh_surf, rh_surf],
           overlays=[lh_over, rh_over],
           vmin=-1.2, vmax=1.2,
           cmap='RdBu_r', avg_method='mean',
           title='Correlation (Z)', colorbar=True,
           output_file='macaque_NMT_correlation_plot.png')

# Plot thresholded correlation map
plot_surf4([lh_surf, rh_surf],
           overlays=[lh_over, rh_over],
           vmin=-1.2, vmax=1.2, threshold=0.3,
           cmap='RdBu_r', avg_method='mean',
           title='Correlation (Z)', colorbar=True,
           output_file='macaque_NMT_correlation_plot_thresholded.png')


elapsed = time.time() - start
print('surfaces rendered in {0:.2f} s'.format(elapsed))
