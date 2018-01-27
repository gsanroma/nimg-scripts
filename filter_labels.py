import argparse
import os
import numpy as np
import nibabel as nib

parser = argparse.ArgumentParser(description='Selects a subset of labels from list of labelmaps')
parser.add_argument("--in_dir", type=str, nargs=1, required=True, help='input directory')
parser.add_argument("--in_suffix", type=str, nargs=1, required=True, help='suffix of input label files')
parser.add_argument("--out_dir", type=str, nargs=1, required=True, help='output directory')
parser.add_argument("--out_suffix", type=str, nargs=1, required=True, help='suffix of output filtered files')
parser.add_argument("--include", type=int, nargs='+', action="append", help="label id or list ids (if list, group using 1st id)")
parser.add_argument("--keep_ids", action="store_true", help="keep label ids of original labels (rather than grouping them using the 1st id)")
parser.add_argument("--map", type=int, nargs=2, action="append", help="map 1st label id to the 2nd (after filtering, if applicable)")

# args = parser.parse_args()
args = parser.parse_args('--in_dir /home/sanromag/DATA/WMH/test_RS/kk/wmh_ana '
                         '--in_suffix _aseg.nii.gz '
                         '--out_dir /home/sanromag/DATA/WMH/test_RS/kk/wmh_ana '
                         '--out_suffix _vent.nii.gz '
                         '--include 43 4 '
                         '--map 43 1 '.split())


files_list = os.listdir(args.in_dir[0])
in_files_list = [f for f in files_list if f.endswith(args.in_suffix[0])]
assert in_files_list, "List of input labels is empty"

out_files_list = [f.split(args.in_suffix[0])[0] + args.out_suffix[0] for f in in_files_list]

for in_file, out_file in zip(in_files_list, out_files_list):

    print("Processing %s" % in_file)

    in_nib = nib.load(os.path.join(args.in_dir[0], in_file))

    in0 = in_nib.get_data()

    out0 = np.zeros(in0.shape, dtype=in0.dtype)

    for labels in args.include:
        for label in labels:
            value = label if args.keep_ids else labels[0]
            out0[np.where(in0 == label)] = value

    if args.map:
        out1 = np.copy(out0)
        for map_pair in args.map:
            out1[np.where(out0 == map_pair[0])] = map_pair[1]

    out_final = out0 if not args.map else out1
    out_nib = nib.Nifti1Image(out_final, in_nib.affine, in_nib.header)
    nib.save(out_nib, os.path.join(args.out_dir[0], out_file))


