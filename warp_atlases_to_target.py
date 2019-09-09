import argparse
import os
import sys

from scheduler import Launcher

parser = argparse.ArgumentParser(description='Warp atlas images in a directory to a target image using given linear and/or deformable transforms.\n'
                                             'Optionally, the target can be warped to atlases with the --inverse flag.')
parser.add_argument("--atlas_dir", type=str, nargs=1, required=True, help='dir of atlas images')
parser.add_argument("--atlas_linear_suffix", type=str, nargs=1, action="append", help="suffix of images to be interpolated with linear")
parser.add_argument("--atlas_nearest_suffix", type=str, nargs=1, action="append", help="suffix of images to be interpolated with nn")
parser.add_argument("--atlas_reg_dir", type=str, nargs=1, required=True, help='directory of transformations from atlas to template')
parser.add_argument("--atlas_linear_intfix", type=str, nargs=1, required=True, help="intfix of the input linear transform")
parser.add_argument("--atlas_deform_intfix", type=str, nargs=1, help="(optional) intfix of the input deformation field")

parser.add_argument("--target_file", type=str, nargs=1, required=True, help='target image file')
parser.add_argument("--target_suffix", type=str, nargs=1, required=True, help="suffix of target (to determine name of transformation file)")
parser.add_argument("--target_reg_dir", type=str, nargs=1, required=True, help='directory of transformations from target to template')
parser.add_argument("--target_linear_intfix", type=str, nargs=1, required=True, help="intfix of the input linear transform")
parser.add_argument("--target_deform_intfix", type=str, nargs=1, help="(optional) intfix of the input deformation field")

parser.add_argument("--out_dir", type=str, nargs=1, required=True, help='directory to store warped files')
parser.add_argument("--out_suffix", type=str, nargs=1, required=True, help="suffix to be added to the filename without extension")
parser.add_argument("--inverse", action="store_true", help="warp target to atlases (beware of interpolation trick!)")
parser.add_argument("--float", action="store_true", help='use single precision computations')
parser.add_argument("--num_procs", type=int, nargs=1, default=[8], help='number of concurrent processes ')

args = parser.parse_args()

if args.atlas_linear_suffix is None:
    args.atlas_linear_suffix = []
if args.atlas_nearest_suffix is None:
    args.atlas_nearest_suffix = []

atlas_files_list = os.listdir(args.atlas_dir[0])
atlas_linear_superlist = [[f for f in atlas_files_list if f.endswith(args.atlas_linear_suffix[i][0])] for i in range(len(args.atlas_linear_suffix))]
atlas_nearest_superlist = [[f for f in atlas_files_list if f.endswith(args.atlas_nearest_suffix[i][0])] for i in range(len(args.atlas_nearest_suffix))]

# create output directory
if not os.path.exists(args.out_dir[0]):
    os.makedirs(args.out_dir[0])

if not args.inverse:
    print "Warping each atlas to target"
else:
    print "Warping target to each atlas"

launcher = Launcher(args.num_procs[0])

antsapplytransforms_path = os.path.join(os.environ['ANTSPATH'], 'antsApplyTransforms')

target_name = os.path.basename(args.target_file[0]).split(args.target_suffix[0])[0]

for atlas_files_list, suffix, interpolation in zip(atlas_linear_superlist, args.atlas_linear_suffix, ['Linear']*len(args.atlas_linear_suffix)) + \
        zip(atlas_nearest_superlist, args.atlas_nearest_suffix, ['NearestNeighbor']*len(args.atlas_nearest_suffix)):

    for atlas_file in atlas_files_list:

        cmdline = [antsapplytransforms_path, '--dimensionality', '3']
        cmdline += ['--interpolation', interpolation]
        cmdline += ['--default-value', '0']
        if args.float:
            cmdline += ['--float']

        atlas_name = atlas_file.split(suffix[0])[0]

        if not args.inverse:

            cmdline += ['--input', os.path.join(args.atlas_dir[0], atlas_file)]
            cmdline += ['--reference-image', args.target_file[0]]

            cmdline += ['--transform', '[{},1]'.format(os.path.join(args.target_reg_dir[0], target_name + args.target_linear_intfix[0] + '0GenericAffine.mat'))]
            if args.target_deform_intfix is not None:
                cmdline += ['--transform', os.path.join(args.target_reg_dir[0], target_name + args.target_deform_intfix[0] + '1InverseWarp.nii.gz')]
            if args.atlas_deform_intfix is not None:
                cmdline += ['--transform', os.path.join(args.atlas_reg_dir[0], atlas_name + args.atlas_deform_intfix[0] + '1Warp.nii.gz')]
            cmdline += ['--transform', os.path.join(args.atlas_reg_dir[0], atlas_name + args.atlas_linear_intfix[0] + '0GenericAffine.mat')]

        else:

            cmdline += ['--input', args.target_file[0]]
            cmdline += ['--reference-image', os.path.join(args.atlas_dir[0], atlas_file)]

            cmdline += ['--transform', '[{},1]'.format(os.path.join(args.atlas_reg_dir[0], atlas_name + args.atlas_linear_intfix[0] + '0GenericAffine.mat'))]
            if args.atlas_deform_intfix is not None:
                cmdline += ['--transform', os.path.join(args.atlas_reg_dir[0], atlas_name + args.atlas_deform_intfix[0] + '1InverseWarp.nii.gz')]
            if args.target_deform_intfix is not None:
                cmdline += ['--transform', os.path.join(args.target_reg_dir[0], target_name + args.target_deform_intfix[0] + '1Warp.nii.gz')]
            cmdline += ['--transform', os.path.join(args.target_reg_dir[0], target_name + args.target_linear_intfix[0] + '0GenericAffine.mat')]

        cmdline += ['--output', os.path.join(args.out_dir[0], atlas_file.split(os.extsep, 1)[0] + args.out_suffix[0])]


        #
        # launch

        job_name = atlas_file.split(os.extsep, 1)[0]
        print "Launching warping of file {}".format(job_name)

        launcher.add(job_name, ' '.join(cmdline), args.out_dir[0])
        launcher.run(job_name)

# Wait for the jobs to finish (in cluster)
print "Waiting for warping jobs to finish..."
launcher.wait()

print "Warping finished."


