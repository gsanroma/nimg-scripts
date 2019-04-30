import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Registers images to template. Can use initial transformation.')
parser.add_argument("--mov_dir", type=str, nargs=1, required=True, help='directory input images')
parser.add_argument("--mov_suffix", type=str, nargs=1, required=True, help='suffix input images')
parser.add_argument("--template_file", type=str, nargs=1, required=True, help='template image')
parser.add_argument("--out_dir", type=str, nargs=1, required=True, help='output directory for transformation files')
parser.add_argument("--out_suffix", type=str, nargs=1, help="(optional) Suffix to be added to the moving image (if none, no warping is done)")
parser.add_argument("--xfm_dir", type=str, nargs=1, help="(optional) directory with transforms")
parser.add_argument("--xfm_suffix", type=str, nargs=1, help="(optional) Suffix of transforms (no optimization)")
parser.add_argument("--num_procs", type=int, nargs=1, default=[8], help='number of concurrent processes ')


args = parser.parse_args()
# args = parser.parse_args(''
#                          '--mov_dir /home/sanromag/DATA/WMH/test_flirt2mni '
#                          '--mov_suffix _t95.nii.gz '
#                          '--template_file /home/sanromag/DATA/WMH/template/MNI152_T1_1mm_brain.nii.gz '
#                          '--out_dir /home/sanromag/DATA/WMH/test_flirt2mni '
#                          '--out_suffix _segmni.nii.gz '
#                          '--xfm_dir /home/sanromag/DATA/WMH/test_flirt2mni '
#                          '--xfm_suffix _flirt.mat '
#                          ''.split())

sys.path.insert(0, os.path.join(os.environ['HOME'], 'CODE', 'src', 'modules'))
from scheduler import Launcher

launcher = Launcher(args.num_procs[0])

#
# Initial checks
#

files_list = os.listdir(args.mov_dir[0])
img_list = [f for f in files_list if f.endswith(args.mov_suffix[0])]
assert img_list, "List of input images is empty"

assert os.path.exists(args.template_file[0]), "Template file not found"

# create output directory
if not os.path.exists(args.out_dir[0]):
    os.makedirs(args.out_dir[0])

#
# Main loop
#

flirt_path = 'flirt'

name_list = []

for img_file in img_list:

    img_path = os.path.join(args.mov_dir[0], img_file)

    cmdline = [flirt_path]
    cmdline.extend(['-in', img_path])
    cmdline.extend(['-ref', args.template_file[0]])
    if args.out_suffix is not None:
        cmdline.extend(['-out', os.path.join(args.out_dir[0], img_file.split(os.extsep, 1)[0] + args.out_suffix[0])])
    cmdline.extend(['-interp', 'trilinear'])
    if args.xfm_suffix is None:
        cmdline.extend(['-omat', os.path.join(args.out_dir[0], img_file.split(os.extsep, 1)[0] + '.mat')])
        cmdline.extend(['-bins', '256'])
        cmdline.extend(['-cost', 'mutualinfo'])
        cmdline.extend(['-searchrx', '-90.0', '90.0'])
        cmdline.extend(['-searchry', '-90.0', '90.0'])
        cmdline.extend(['-searchrz', '-90.0', '90.0'])
        # cmdline.extend(['-2D'])
        cmdline.extend(['-dof', '12'])
    else:
        cmdline.extend(['-applyxfm'])
        cmdline.extend(['-init', os.path.join(args.xfm_dir[0], img_file.split(args.mov_suffix[0])[0] + args.xfm_suffix[0])])

    #
    # launch

    print "Launching registration of file {}".format(img_file)

    name_list.append(img_file.split(os.extsep, 1)[0])
    launcher.add(name_list[-1], ' '.join(cmdline), args.out_dir[0])
    launcher.run(name_list[-1])

print "Waiting for registration jobs to finish..."

launcher.wait()

print "Registration finished."

