import time
import numpy as np
from matplotlib import pyplot as plt
from nilearn.surface import vol_to_surf
from brain4views import plot_surf4

start = time.time()

path = '/home/nsirmpilatze/BS/Meshes/katja_surf/'

lh_mesh = path + 'lh.pial.surf.gii'
rh_mesh = path + 'rh.pial.surf.gii'
stat = path + 'BS_Zmap_mean_NMT.nii.gz'
lh_over = vol_to_surf(stat, lh_mesh, radius=1, kind='ball')
rh_over = vol_to_surf(stat, rh_mesh, radius=1, kind='ball')
lh_surf = path + 'lh.pial_semi_inflated.surf.gii'
rh_surf = path + 'rh.pial_semi_inflated.surf.gii'
lh_sulc = path + 'lh.pial.mean_curv.shape.gii'
rh_sulc = path + 'rh.pial.mean_curv.shape.gii'

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
           sulc_maps=[lh_sulc, rh_sulc],
           vmin=-1.2, vmax=1.2, threshold=0.3,
           cmap='RdBu_r', avg_method='mean',
           title='Correlation (Z)', colorbar=True,
           output_file='macaque_NMT_correlation_plot_thresholded.png')


elapsed = time.time() - start
print('surfaces rendered in {0:.2f} s'.format(elapsed))
