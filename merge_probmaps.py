import argparse
import os
import numpy as np
import nibabel as nib
from copy import copy

parser = argparse.ArgumentParser(description='merge probability maps')
parser.add_argument("--in_dir_list", type=str, nargs='+', required=True, help='list of input directories')
parser.add_argument("--in_suffix", type=str, nargs=1, required=True, help='suffix of input probmaps to be merged (same suffix in all dirs)')
parser.add_argument("--out_dir", type=str, nargs=1, required=True, help='output dir where to save merged results to')
parser.add_argument("--keep_probmaps", action='store_true', help='keeps averaged probability maps (with original suffix)')
parser.add_argument("--threshold", type=float, nargs=1, default=[0.5], help='binarization threshold (default: 0.5)')
parser.add_argument("--out_suffix", type=str, nargs=1, required=True, help='output suffix')

args = parser.parse_args()

# get files and names from 1st directory in list
files_list = os.listdir(args.in_dir_list[0])
names_list = [f.split(args.in_suffix[0])[0] for f in files_list if f.endswith(args.in_suffix[0])]

# if dir doesn't exist, create
if not os.path.exists(args.out_dir[0]):
    os.makedirs(args.out_dir[0])

# main loop

for file, name in zip(files_list, names_list):

    print("Processing %s" % file)

    ref_nib = None
    avg = None  # average probmap

    # average probmaps in across folders

    for dir in args.in_dir_list:

        prob_nib = nib.load(os.path.join(dir, file))
        prob = prob_nib.get_data()

        if avg is None:
            avg = copy(prob) / float(len(args.in_dirs_list))
            ref_nib = copy(prob_nib)
        else:
            assert prob_nib.header.get_data_shape() == ref_nib.header.get_data_shape(), 'non-equal probmap shapes'
            avg += prob / float(len(args.in_dirs_list))

    # threshold average
    seg = np.zeros(avg.shape, dtype=np.uint8)
    seg[avg > args.threshold[0]] = 1

    # save
    seg_nib = nib.Nifti1Image(seg, ref_nib.affine, ref_nib.header)
    seg_nib.set_data_dtype(np.uint8)
    nib.save(seg_nib, os.path.join(args.out_dir[0], name + args.out_suffix[0]))

    # save probmaps
    if args.keeprobmaps:
        avg_nib = nib.Nifti1Image(avg, ref_nib.affine, ref_nib.header)
        nib.save(avg_nib, os.path.join(args.out_dir[0], file))


