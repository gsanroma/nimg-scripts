import argparse
import os
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Selects a subset of labels from list of labelmaps')
parser.add_argument("--in_dir", type=str, nargs=1, required=True, help='input directory')
parser.add_argument("--in_suffix", type=str, nargs=1, required=True, help='suffix of input label files')
parser.add_argument("--out_dir", type=str, nargs=1, required=True, help='output directory')
parser.add_argument("--out_suffix", type=str, nargs=1, required=True, help='suffix of output filtered files')
parser.add_argument("--include", type=int, nargs='+', action="append", help="label id or list ids (if list, group using 1st id)")
parser.add_argument("--include_range", type=int, nargs=2, help="min and max of range of label ids to keep (grouped using min)")
parser.add_argument("--keep_ids", action="store_true", help="keep label ids of original labels (rather than grouping them using the 1st id)")
parser.add_argument("--map", type=int, nargs=2, action="append", help="map 1st label id to the 2nd (after filtering, if applicable)")
parser.add_argument("--fixed_id", type=int, nargs=1, help="assign a fixed id to all labels")
parser.add_argument("--num_procs", type=int, nargs=1, default=[8], help='number of concurrent processes ')

args = parser.parse_args()
# args = parser.parse_args(''
#                          '--in_dir /home/sanromag/DATA/tmp '
#                          '--in_suffix _FSwmparc.nii.gz '
#                          '--out_dir /home/sanromag/DATA/tmp '
#                          '--out_suffix _filtered.nii.gz '
#                          '--include_range 3000 4999 '
#                          '--keep_ids '
#                          ''.split())

files_list = os.listdir(args.in_dir[0])
in_files_list = [f for f in files_list if f.endswith(args.in_suffix[0])]
assert in_files_list, "List of input labels is empty"

out_files_list = [f.split(args.in_suffix[0])[0] + args.out_suffix[0] for f in in_files_list]

# create output directory
if not os.path.exists(args.out_dir[0]):
    os.makedirs(args.out_dir[0])

def filt_lab(in_path, out_path, args):
    """Filters labels

    Parameters
    ----------
    in_path : string
        path of the input file
    out_path : string
        path of the output file
    """
# for in_file, out_file in zip(in_files_list, out_files_list):

    print("Processing %s" % os.path.basename(in_path))

    in_nib = nib.load(in_path)

    in0 = in_nib.get_data()

    out0 = np.zeros(in0.shape, dtype=in0.dtype)

    if args.include is not None:
        for labels in args.include:
            for label in labels:
                value = labels[0]
                if args.keep_ids: value = label
                if args.fixed_id is not None: value = args.fixed_id[0]
                out0[in0 == label] = value

    elif args.include_range is not None:
        selection_mask = np.logical_and(in0 >= args.include_range[0], in0 <= args.include_range[1])
        if args.fixed_id is not None:
            out0[selection_mask] = args.fixed_id[0]
        elif args.keep_ids:
            out0[selection_mask] = in0[selection_mask]
        else:
            out0[selection_mask] = args.include_range[0]

    if args.map is not None:
        out1 = np.copy(out0)
        for map_pair in args.map:
            out1[out0 == map_pair[0]] = map_pair[1]

    out_final = out0 if not args.map else out1
    out_nib = nib.Nifti1Image(out_final, in_nib.affine, in_nib.header)
    nib.save(out_nib, out_path)


Parallel(n_jobs=args.num_procs[0])(delayed(filt_lab)(os.path.join(args.in_dir[0], in_file), os.path.join(args.out_dir[0], out_file), args) for in_file, out_file in zip(in_files_list, out_files_list))

