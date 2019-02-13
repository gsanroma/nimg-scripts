import argparse
import os
import numpy as np
import nibabel as nib
from scipy.spatial.distance import dice

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Plots lesion vol vs. dice.')
parser.add_argument("--est_dir", type=str, nargs=1, required=True, help="Directory of estimated segmentations")
parser.add_argument("--est_suffix", type=str, nargs=1, required=True, help="Suffix of estimated segmentation files")
parser.add_argument("--gtr_dir", type=str, nargs=1, required=True, help="Directory of ground-truth segmentations")
parser.add_argument("--gtr_suffix", type=str, nargs=1, required=True, help="Suffix of ground truth segmentation files")
parser.add_argument("--brain_dir", type=str, nargs=1, required=True, help="Directory of brain mask (to normalize lesion load)")
parser.add_argument("--brain_suffix", type=str, nargs=1, required=True, help="Suffix of brain mask files")
parser.add_argument("--out_fig", type=str, nargs=1, help="Output fig")

# args = parser.parse_args()
args = parser.parse_args('--est_dir /home/sanromag/DATA/WMH/BIANCA/batch_run_recomm/locpts2/ '
                         '--est_suffix _t95.nii.gz '
                         '--gtr_dir /home/sanromag/DATA/WMH/RS/data_proc '
                         '--gtr_suffix _WMHmaskbin.nii.gz '
                         '--brain_dir /home/sanromag/DATA/WMH/RS/data_proc '
                         '--brain_suffix _brainmaskWarped.nii.gz '
                         '--out_fig /home/sanromag/DATA/WMH/BIANCA/vol_dice.png '.split())

# List of estimated files

files_list = os.listdir(args.est_dir[0])
est_files = [f for f in files_list if f.endswith(args.est_suffix[0])]
est_names = [f.split(args.est_suffix[0])[0] for f in est_files]
assert est_files, "No estimated segmentation found"

# List of ground truth files

gtr_files = [f + args.gtr_suffix[0] for f in est_names]
assert not False in [os.path.exists(os.path.join(args.gtr_dir[0], f)) for f in gtr_files], "Some ground-truth segmentations not found"

# List of brain masks

brain_files = [f + args.brain_suffix[0] for f in est_names]
assert not False in [os.path.exists(os.path.join(args.brain_dir[0], f)) for f in brain_files], "Some brain mask file not found"


#
# Read actual files

est_vols = np.zeros(len(est_names))
gtr_vols = np.zeros(len(est_names))
brain_vols = np.zeros(len(est_names))

dices = np.zeros(len(est_names))

for i, (est_file, gtr_file) in enumerate(zip(est_files, gtr_files)):

    print('Reading %s' % est_names[i])

    est_nib = nib.load(os.path.join(args.est_dir[0], est_file))
    est = est_nib.get_data().astype(np.bool)
    est_vols[i] = est.sum()

    gtr_nib = nib.load(os.path.join(args.gtr_dir[0], gtr_file))
    gtr = gtr_nib.get_data().astype(np.bool)
    gtr_vols[i] = gtr.sum()

    dices[i] = 1. - dice(est.ravel(), gtr.ravel())

    brain_nib = nib.load(os.path.join(args.brain_dir[0], brain_files[i]))
    brain = brain_nib.get_data().astype(np.bool)
    brain_vols[i] = brain.sum()

gtr_vol_norm = gtr_vols/brain_vols

# print ids ordered by volume
Idx = np.argsort(gtr_vol_norm)
print('Sorted IDs:')
for i in Idx:
    print('%s: %f' % (est_names[i], gtr_vol_norm[i]))

# plot
fig = plt.figure()

plt.scatter(gtr_vol_norm, dices)

plt.xlabel('Lesion volume')
plt.ylabel('Dice score')

plt.savefig(args.out_fig[0])

