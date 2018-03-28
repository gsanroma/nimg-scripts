import argparse
import os
import numpy as np
import nibabel as nib

parser = argparse.ArgumentParser(description='Thresholds label probability maps and outputs binary segmentations')
parser.add_argument("--in_dir", type=str, nargs=1, required=True, help='input directory')
parser.add_argument("--in_suffix", type=str, nargs=1, required=True, help='suffix of input label probability files')
parser.add_argument("--out_dir", type=str, nargs=1, help='output directory (same as input if not defined)')
parser.add_argument("--out_suffix_list", type=str, nargs='+', required=True, help='list of suffixes of output thresholded labelmap files')
parser.add_argument("--threshold_list", type=float, nargs='+', required=True, help="List of thresholds")

args = parser.parse_args()
# args = parser.parse_args('--in_dir /home/sanromag/DATA/WMH/train_RS/kk '
#                          '--in_suffix _brainmaskWarped.nii.gz '
#                          '--out_dir /home/sanromag/DATA/WMH/train_RS/kk '
#                          '--out_suffix_list _brainmaskWarped.nii.gz '
#                          '--threshold_list 0.1 '.split())

assert len(args.out_suffix_list) == len(args.threshold_list), "same number of thresholds as suffixes should be given"

# set out_dir
out_dir = args.in_dir[0] if args.out_dir is None else args.out_dir[0]
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

files_list = os.listdir(args.in_dir[0])
in_files_list = [f for f in files_list if f.endswith(args.in_suffix[0])]
assert in_files_list, "List of input labels is empty"

for in_file in in_files_list:

    print("Processing %s" % in_file)

    prlab_nib = nib.load(os.path.join(args.in_dir[0], in_file))
    prlab = prlab_nib.get_data()

    for threshold, out_suffix in zip(args.threshold_list, args.out_suffix_list):

        # output mask
        mask = np.zeros(prlab.shape, dtype=np.uint8)
        mask[prlab > threshold] = 1

        # save
        mask_nib = nib.Nifti1Image(mask, prlab_nib.affine, prlab_nib.header)
        mask_nib.set_data_dtype(np.uint8)
        nib.save(mask_nib, os.path.join(out_dir, in_file.split(args.in_suffix[0])[0] + out_suffix))

        del mask


