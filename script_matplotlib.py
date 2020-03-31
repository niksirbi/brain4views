import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib_surface_plotting import plot_surf, plot_surf_rois, plot_surf_parcellation

start = time.time()

# freesurfer subjects direvtory
SUBJECTS_DIR = '/home/nsirmpilatze/FS'
# subject name
SUBJ = 'fsaverage'
# path to freesurfer surface of that subject
surf_dir = '{0}/{1}/surf/'.format(SUBJECTS_DIR, SUBJ)
# path to freesurfer lables of that subject
label_dir = '{0}/{1}/label/'.format(SUBJECTS_DIR, SUBJ)

# Needed meshes and data from freesurfer directories
lh_surf = surf_dir + 'lh.inflated'
rh_surf = surf_dir + 'rh.inflated'
lh_sulc = surf_dir + 'lh.sulc'
rh_sulc = surf_dir + 'rh.sulc'
lh_mask = label_dir + 'lh.cortex.label'
rh_mask = label_dir + 'rh.cortex.label'

'''
# Plot correlation map
lh_over = '/home/nsirmpilatze/BS/Meshes/human_maps/BS_Zmap_mean_lh.surf.gii'
rh_over = '/home/nsirmpilatze/BS/Meshes/human_maps/BS_Zmap_mean_rh.surf.gii'

plot_surf([lh_surf, rh_surf],
          overlays=[lh_over, rh_over],
          sulc_maps=None,
          ctx_masks=[lh_mask, rh_mask],
          vmin=-1.5, threshold=None, vmax=1.5,
          cmap='RdBu_r', avg_method='mean',
          title='Correlation (Z)', colorbar=True,
          output_file='human_correlation_plot.png')

elapsed = time.time() - start
print('surfaces rendered in {0:.2f} s'.format(elapsed))


# Plot parcellation
lh_parc = label_dir + 'lh.aparc.a2009s.annot'
rh_parc = label_dir + 'rh.aparc.a2009s.annot'

plot_surf_parcellation(
          [lh_surf, rh_surf],
          [lh_parc, rh_parc],
          sulc_maps=None,
          ctx_masks=[lh_mask, rh_mask],
          cmap='rainbow', shuffle_cmap=True,
          title='Destrieux Atlas',
          output_file='human_destrieux_atlas.png')

elapsed = time.time() - start
print('surfaces rendered in {0:.2f} s'.format(elapsed))
'''

# Plot ROIs
lh_roi = label_dir + 'lh.V1_exvivo.label'
rh_roi = label_dir + 'rh.V1_exvivo.label'

plot_surf_rois([lh_surf, rh_surf],
               [lh_roi, rh_roi],
               sulc_maps=[lh_sulc, rh_sulc],
               color='Red')