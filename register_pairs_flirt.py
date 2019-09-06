import argparse
import os
import sys

parser = argparse.ArgumentParser('Register pairs of images (eg, baseline and follow-up) using FSL flirt')
parser.add_argument("--ref_dir", type=str, nargs=1, required=True, help="Directory containing the reference images")
parser.add_argument("--ref_suffix", type=str, nargs=1, required=True, help="Suffix of reference images")
parser.add_argument("--mov_dir", type=str, nargs=1, required=True, help="Directory containing the moving images")
parser.add_argument("--mov_suffix_list", type=str, nargs='+', required=True, help="List of suffixes of moving images (must have same prefix as reference)")
parser.add_argument("--interp", type=str, nargs=1, default=['trilinear'], help="Interpolation: [trilinear,nearestneighbour,sinc,spline]")
parser.add_argument("--out_dir", type=str, nargs=1, required=True, help="Where to store the results to")
parser.add_argument("--out_suffix", type=str, nargs=1, required=True, help="Suffix to be added to the moving image w/o extension")
parser.add_argument("--xfm_dir", type=str, nargs=1, help="(optional) directory with transforms")
parser.add_argument("--xfm_suffix_list", type=str, nargs='+', help="(optional) List of suffixes of transforms (no optimization. If only 1, applied to all moving imgs.)")
parser.add_argument("--searchxyz", type=float, nargs=1, default=[25.], help='search degrees in x, y, z axes (default 25) ')
parser.add_argument("--num_procs", type=int, nargs=1, default=[ ], help='number of concurrent processes ')


args = parser.parse_args()
# args = parser.parse_args(''
#                          '--ref_dir /home/sanromag/DATA/WMH/train_RS/data_proc '
#                          '--ref_suffix _FLAIR.nii.gz '
#                          '--mov_dir /home/sanromag/DATA/WMH/train_RS/data '
#                          '--mov_suffix_list _brainmask.nii.gz '
#                          '--interp trilinear '
#                          '--out_dir /home/sanromag/DATA/WMH/train_RS/kk '
#                          '--out_suffix Warped.nii.gz '
#                          '--xfm_dir /home/sanromag/DATA/WMH/train_RS/transform_pairs '
#                          '--xfm_suffix_list _T1FS.mat '
#                          '--num_procs 30 '
#                          ''.split())


sys.path.insert(0, os.path.join(os.environ['HOME'], 'CODE', 'src', 'modules'))
from scheduler import Launcher

launcher = Launcher(args.num_procs[0])

#
# Initial checks
#

if args.xfm_suffix_list is not None:
    assert len(args.xfm_suffix_list) == 1 or len(args.xfm_suffix_list) == len(args.mov_suffix_list), "Must have only one transform OR the same number as moving images"

files_list = os.listdir(args.ref_dir[0])
ref_list = [f for f in files_list if f.endswith(args.ref_suffix[0])]
assert ref_list, "List of input images is empty"

# create output directory
if not os.path.exists(args.out_dir[0]):
    os.makedirs(args.out_dir[0])

#
# Main program
#

# separate ids and dates
id_list = [f.split(args.ref_suffix[0])[0] for f in ref_list]

flirt_path = os.path.join(os.environ['FSLDIR'], 'bin', 'flirt')

name_list = []

for id, ref in zip(id_list, ref_list):

    for i, suffix in enumerate(args.mov_suffix_list):

        cmdline = [flirt_path]
        cmdline.extend(['-in', os.path.join(args.mov_dir[0], id + suffix)])
        cmdline.extend(['-ref', os.path.join(args.ref_dir[0], ref)])
        cmdline.extend(['-out', os.path.join(args.out_dir[0], id + suffix.split(os.extsep, 1)[0] + args.out_suffix[0])])
        cmdline.extend(['-interp', args.interp[0]])
        if args.xfm_suffix_list is None:
            cmdline.extend(['-omat', os.path.join(args.out_dir[0], id + suffix.split(os.extsep, 1)[0] + '.mat')])
            cmdline.extend(['-bins', '256'])
            cmdline.extend(['-cost', 'mutualinfo'])
            cmdline.extend(['-searchrx', '-%0.2f' % args.searchxyz[0], '%0.2f' % args.searchxyz[0]])
            cmdline.extend(['-searchry', '-%0.2f' % args.searchxyz[0], '%0.2f' % args.searchxyz[0]])
            cmdline.extend(['-searchrz', '-%0.2f' % args.searchxyz[0], '%0.2f' % args.searchxyz[0]])
            cmdline.extend(['-2D'])
            cmdline.extend(['-dof', '12'])
        else:
            cmdline.extend(['-applyxfm'])
            if len(args.xfm_suffix_list) == 1: xfm_suffix = args.xfm_suffix_list[0]
            else: xfm_suffix = args.xfm_suffix_list[i]
            cmdline.extend(['-init', os.path.join(args.xfm_dir[0], id + xfm_suffix)])

        # launch

        print "Launching registration of subject %s" % (id)

        name_list.append(id + suffix.split(os.extsep, 1)[0])
        launcher.add(name_list[-1], ' '.join(cmdline), args.out_dir[0])
        launcher.run(name_list[-1])

print "Waiting for registration jobs to finish..."

launcher.wait()

print "Registration finished."



