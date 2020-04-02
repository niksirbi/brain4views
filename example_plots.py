import time
import numpy as np
from matplotlib import pyplot as plt
from brain4views import plot_surf4, plot_surf4_parcellation

start = time.time()

# freesurfer subjects direvtory
SUBJECTS_DIR = '/home/nsirmpilatze/FS'
# subject name
SUBJ = 'fsaverage'
# path to freesurfer surface of that subject
surf_dir = '{0}/{1}/surf/'.format(SUBJECTS_DIR, SUBJ)
# path to freesurfer lables of that subject
label_dir = '{0}/{1}/label/'.format(SUBJECTS_DIR, SUBJ)

# surface meshes (.inflated, .pial, .white etc.)
lh_surf = surf_dir + 'lh.inflated'
rh_surf = surf_dir + 'rh.inflated'
# sulcal depth maps (.sulc)
lh_sulc = surf_dir + 'lh.sulc'
rh_sulc = surf_dir + 'rh.sulc'
# cortical thickness maps (.thickness)
lh_thick = surf_dir + 'lh.thickness'
rh_thick = surf_dir + 'rh.thickness'
# cortical masks (or other .label files)
lh_mask = label_dir + 'lh.cortex.label'
rh_mask = label_dir + 'rh.cortex.label'
# cortical parcellations - Destrieux atlas (or other .annot files)
lh_parc = label_dir + 'lh.aparc.a2009s.annot'
rh_parc = label_dir + 'rh.aparc.a2009s.annot'

# Plot sulcal map only
plot_surf4([lh_surf, rh_surf],
           overlays=None,
           sulc_maps=[lh_sulc, rh_sulc],
           ctx_masks=[lh_mask, rh_mask],
           output_file='human_sulcal_plot.png')
'''
# plot cortical thickness
plot_surf4([lh_surf, rh_surf],
           overlays=[lh_thick, rh_thick],
           sulc_maps=None,
           ctx_masks=[lh_mask, rh_mask],
           vmin=1, threshold=None, vmax=4,
           cmap='inferno', avg_method='mean',
           title='Cortical thickness (mm)', colorbar=True,
           output_file='human_cortical_thickness.png')

# Plot correlation map
lh_over = '/home/nsirmpilatze/BS/Meshes/human_maps/BS_Zmap_mean_lh.surf.gii'
rh_over = '/home/nsirmpilatze/BS/Meshes/human_maps/BS_Zmap_mean_rh.surf.gii'

plot_surf4([lh_surf, rh_surf],
           overlays=[lh_over, rh_over],
           sulc_maps=[lh_sulc, rh_sulc],
           ctx_masks=[lh_mask, rh_mask],
           vmin=-1.5, threshold=0.5, vmax=1.5,
           cmap='RdBu_r', avg_method='mean',
           title='Correlation (Z)', colorbar=True,
           output_file='human_correlation_plot.png')

# Plot parcellation
lh_parc = label_dir + 'lh.aparc.a2009s.annot'
rh_parc = label_dir + 'rh.aparc.a2009s.annot'

plot_surf4_parcellation(
          [lh_surf, rh_surf],
          [lh_parc, rh_parc],
          sulc_maps=None,
          ctx_masks=[lh_mask, rh_mask],
          cmap='gist_rainbow', shuffle_cmap=True,
          title='Destrieux Atlas',
          output_file='human_destrieux_atlas.png')

elapsed = time.time() - start
print('surfaces rendered in {0:.2f} s'.format(elapsed))
'''