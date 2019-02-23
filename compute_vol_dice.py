import argparse
import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.spatial.distance import dice

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# import seaborn as sns


parser = argparse.ArgumentParser(description='Plots lesion vol vs. dice.')
parser.add_argument("--ref_dir", type=str, nargs=1, required=True, help="Directory of reference segmentations"
                                                                        "(used to get the names in case 2nd segmentations are given)")
parser.add_argument("--ref_suffix", type=str, nargs=1, required=True, help="Suffix of reference segmentations")
parser.add_argument("--in2_dir", type=str, nargs=1, help="(optional) Directory of second segmentations")
parser.add_argument("--in2_suffix", type=str, nargs=1, help="(optional) Suffix of second segmentations")
parser.add_argument("--per_label", action='store_true', help="extract values per label")
# parser.add_argument("--brain_dir", type=str, nargs=1, help="(optional) Directory of brain mask (to normalize lesion load)")
# parser.add_argument("--brain_suffix", type=str, nargs=1, help="(optional) Suffix of brain mask files")
parser.add_argument("--out_fig", type=str, nargs=1, help="Output fig")
parser.add_argument("--out_csv", type=str, nargs=1, help="Output csv")

args = parser.parse_args()
# args = parser.parse_args('--ref_dir /home/sanromag/DATA/OB/tmp/ '
#                          '--ref_suffix _Segm.nii.gz '
#                          '--in2_dir /home/sanromag/DATA/OB/data_n4_norm '
#                          '--in2_suffix _OBV.nii.gz '
#                          '--per_label '
#                          '--out_csv /home/sanromag/DATA/OB/tmp/kk.csv '
#                          ''.split())

# List of estimated files

files_list = os.listdir(args.ref_dir[0])
ref_files = [f for f in files_list if f.endswith(args.ref_suffix[0])]
ref_names = [f.split(args.ref_suffix[0])[0] for f in ref_files]
assert ref_files, "No reference segmentation found"

# List of ground truth files
in2_files = []
if args.in2_dir is not None:
    in2_files = [f + args.in2_suffix[0] for f in ref_names]
    assert not False in [os.path.exists(os.path.join(args.in2_dir[0], f)) for f in in2_files], "Some second segmentation not found"

# # List of brain masks
# brain_files = []
# if args.brain_dir is not None:
#     brain_files = [f + args.brain_suffix[0] for f in est_names]
#     assert not False in [os.path.exists(os.path.join(args.brain_dir[0], f)) for f in brain_files], "Some brain mask file not found"


#
# Read actual files

df = pd.DataFrame([])

for i, (ref_name, ref_file) in enumerate(zip(ref_names, ref_files)):

    print('Reading %s (%d of %d)' % (ref_names[i], i, len(ref_names)))

    ref = np.round(nib.load(os.path.join(args.ref_dir[0], ref_file)).get_data()).astype(int)

    in2 = None
    if in2_files:
        in2 = np.round(nib.load(os.path.join(args.in2_dir[0], in2_files[i])).get_data()).astype(int)

    # compute per-label metrics
    if args.per_label:

        u_lab = np.unique(ref)
        u_lab = np.delete(u_lab, np.where(u_lab == 0))

        for lab in u_lab:
            df.loc[ref_name, 'ref_vol_lab%d' % lab] = (ref == lab).sum()
            if in2_files:
                df.loc[ref_name, 'in2_vol_lab%d' % lab] = (in2 == lab).sum()
                df.loc[ref_name, 'dice_lab%d' % lab] = 1. - dice((ref == lab).ravel(), (in2 == lab).ravel())

    # compute whole volume metrics
    df.loc[ref_name, 'ref_vol'] = (ref > 0).sum()
    df.loc[ref_name, 'in2_vol'] = (in2 > 0).sum()

    df.loc[ref_name, 'dice'] = 1. - dice((ref > 0).ravel(), (in2 > 0).ravel())

    # brain_nib = nib.load(os.path.join(args.brain_dir[0], brain_files[i]))
    # brain = brain_nib.get_data().astype(np.bool)
    # brain_vols[i] = brain.sum()

# gtr_vol_norm = gtr_vols/brain_vols

# plot
if args.out_fig is not None and args.in2_dir is not None:

    fig = plt.figure()

    plt.scatter(df['ref_vol'].values, df['dice'].values)

    plt.xlabel('volume')
    plt.ylabel('dice')

    plt.savefig(args.out_fig[0])

# save csv
if args.out_csv is not None:

    df.to_csv(args.out_csv[0])