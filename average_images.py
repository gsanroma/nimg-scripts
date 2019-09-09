import argparse
import os
import sys
import numpy as np
import nibabel as nib

parser = argparse.ArgumentParser(description='Average images')
parser.add_argument("--in_dir", type=str, nargs=1, required=True, help='input directory')
parser.add_argument("--in_suffix", type=str, nargs=1, required=True, help='suffix of input images')
parser.add_argument("--out_avg", type=str, nargs=1, help='output averaged image')

args = parser.parse_args()

from utils import get_files_superlist

# get file list
names_list, files_superlist, _ = get_files_superlist(args.in_dir, args.in_suffix)
files_list = files_superlist[0]
in_dir = args.in_dir[0]

avg_img = None  # average image
ref_nib = None  # reference nibabel image

for in_file in files_list:

    print("Processing %s" % in_file)

    img_nib = nib.load(os.path.join(args.in_dir[0], in_file))
    img = img_nib.get_data()

    if avg_img is None:
        avg_img = np.zeros(img.shape)
        ref_nib = img_nib

    assert img.shape == avg_img.shape, "all images must be of same size"

    avg_img += img / float(len(names_list))

# save
avg_nib = nib.Nifti1Image(avg_img, ref_nib.affine, ref_nib.header)
nib.save(avg_nib, args.out_avg[0])



