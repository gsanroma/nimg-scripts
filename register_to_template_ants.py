import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Registers images to template. Can use initial transformation.')
parser.add_argument("--in_dir", type=str, nargs=1, required=True, help='directory input images')
parser.add_argument("--img_suffix", type=str, nargs=1, required=True, help='suffix input images')
parser.add_argument("--template_file", type=str, nargs=1, required=True, help='template image')
parser.add_argument("--template_mask", type=str, nargs=1, help="(optional) to limit registration to a region (better start with good initialization)")
parser.add_argument("--init_warp_dir_suffix", type=str, nargs='+', action="append", help="(optional) dir, suffix (and inverse flag for affine) of warps to be used as initialization (in order)")
parser.add_argument("--transform", type=str, nargs=2, required=True, help="Rigid[*] | Affine[*] | Syn[*], 1<=resolution<=4 (with \'*\' do all lower resolutions too)")
parser.add_argument("--speedup", type=float, nargs=1, default=[1.], help="Divide default number of iterations by this factor to speedup")
parser.add_argument("--out_warp_intfix", type=str, nargs=1, required=True, help="intfix for output warps")
parser.add_argument("--out_dir", type=str, nargs=1, required=True, help='output directory for transformation files')
parser.add_argument("--output_warped_image", action="store_true", help="output warped images (image name w/o ext + intfix + Warped.nii.gz)")
parser.add_argument("--float", action="store_true", help='use single precision computations')
parser.add_argument("--use_labels", type=str, nargs='+', help='use labels for registration: label_dir, label_suffix, template_labels, [weights_list_for_each_stage]')
parser.add_argument("--num_procs", type=int, nargs=1, default=[8], help='number of concurrent processes ')
parser.add_argument("--num_itk_threads", type=int, nargs=1, default=[1], help='number of threads per ANTs proc ')


args = parser.parse_args()
# args = parser.parse_args('--in_dir /Users/gsanroma/DATA/DATABASES/ADNI/atlases/kk --img_suffix moving.nii.gz --template_file /Users/gsanroma/DATA/DATABASES/ADNI/atlases/kk/fixed.nii.gz --transform Affine 4 --out_warp_intfix rigid3 --out_dir /Users/gsanroma/DATA/DATABASES/ADNI/atlases/kk --init_warp_dir_suffix /Users/gsanroma/DATA/DATABASES/ADNI/atlases/kk moving_warped0GenericAffine.mat --init_warp_dir_suffix /Users/gsanroma/DATA/DATABASES/ADNI/atlases/kk moving_masked_warped20GenericAffine.mat 1 --template_mask /Users/gsanroma/DATA/DATABASES/ADNI/atlases/kk/left2.nii.gz'.split())

from scheduler import Launcher

launcher = Launcher(args.num_procs[0])

# default num of iterations
ITS_LINEAR = ['%d' % int(500 / args.speedup[0]), '%d' % int(250 / args.speedup[0]), '%d' % int(125 / args.speedup[0]), '%d' % int(50 / args.speedup[0])]
ITS_SYN = ['%d' % int(70 / args.speedup[0]), '%d' % int(50 / args.speedup[0]), '%d' % int(30 / args.speedup[0]), '%d' % int(10 / args.speedup[0])]

#
# Initial checks
#

files_list = os.listdir(args.in_dir[0])
img_list = [f for f in files_list if f.endswith(args.img_suffix[0])]
assert img_list, "List of input images is empty"

assert os.path.exists(args.template_file[0]), "Template file not found"
if args.template_mask is not None:
    assert os.path.exists(args.template_mask[0]), "Template mask not found"

if args.use_labels is not None:
    lab_list = [f.split(args.img_suffix[0])[0] + args.use_labels[1] for f in img_list]
    assert False not in [os.path.exists(os.path.join(args.use_labels[0], f)) for f in lab_list], "label files not found"
    assert os.path.exists(args.use_labels[2]), "Template labels not found"

resolution = int(args.transform[1])
assert resolution > 0 and resolution < 5, "Wrong resolution"

# create output directory
if not os.path.exists(args.out_dir[0]):
    os.makedirs(args.out_dir[0])

#
# Main loop
#

antsregistration_path = os.path.join(os.environ['ANTSPATH'], 'antsRegistration')

name_list = []

for img_file in img_list:

    img_name = img_file.split(args.img_suffix[0])[0]
    img_path = os.path.join(args.in_dir[0], img_file)

    if args.use_labels is not None:
        lab_path = os.path.join(args.use_labels[0], img_name + args.use_labels[1])
    weight_idx = 3

    cmdline = ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d' % args.num_itk_threads[0], antsregistration_path, '--dimensionality', '3']

    out_prefix = os.path.join(args.out_dir[0], img_file.split(os.extsep, 1)[0] + args.out_warp_intfix[0])
    if args.output_warped_image:
        cmdline.extend(['--output', '[%s,%sWarped.nii.gz]' % (out_prefix, out_prefix)])
    else:
        cmdline.extend(['--output', '%s' % (out_prefix)])
    cmdline.extend(['--write-composite-transform', '0'])
    cmdline.extend(['--collapse-output-transforms', '1'])
    cmdline.extend(['--initialize-transforms-per-stage', '0'])
    cmdline.extend(['--interpolation', 'Linear'])
    if args.float:
        cmdline.extend(['--float', '1'])

    #
    # init transforms

    if not args.init_warp_dir_suffix:
        cmdline.extend(['--initial-moving-transform', '[{},{},1]'.format(args.template_file[0], img_path)])
    else:
        for init_warp in args.init_warp_dir_suffix[::-1]:
            if len(init_warp) < 3:
                cmdline.extend(['--initial-moving-transform', os.path.join(init_warp[0], img_name + init_warp[1])])
            else:
                cmdline.extend(['--initial-moving-transform', '[{},{}]'.format(os.path.join(init_warp[0], img_name + init_warp[1]), init_warp[2])])

    #
    # transforms

    if args.transform[0][-1] == '*':  # use all resolutions
        its_linear = ITS_LINEAR[:resolution] + ['0']*(4-resolution)
        its_syn = ITS_SYN[:resolution] + ['0']*(4-resolution)
    else:  # use only specified resolution
        its_linear = ['0']*(resolution - 1) + [ITS_LINEAR[resolution - 1]] + ['0'] * (4 - resolution)
        its_syn = ['0']*(resolution - 1) + [ITS_SYN[resolution - 1]] + ['0'] * (4 - resolution)

    smooth_sig = '4x2x1x0'
    shrink_fac = '8x4x2x1'

    if args.transform[0].rstrip('*') == 'Rigid' or not args.init_warp_dir_suffix:

        cmdline.extend(['--transform', 'Rigid[0.1]'])

        w_img, w_lab = 1.0, 0.0
        if args.use_labels is not None:
            if len(args.use_labels) > weight_idx:
                w_lab = float(args.use_labels[weight_idx])
            w_img = 1.0 - w_lab
            weight_idx += 1

        if not args.template_mask:
            cmdline.extend(['--metric', 'MI[{},{},{},32,Regular,0.25]'.format(args.template_file[0], img_path, w_img)])
        else:
            cmdline.extend(['--metric', 'GC[{},{},{}]'.format(args.template_file[0], img_path, w_img)])

        if w_lab > 0.0:
            cmdline.extend(['--metric', 'MeanSquares[{},{},{}]'.format(args.use_labels[2], lab_path, w_lab)])

        cmdline.extend(['--convergence', '[{},1e-8,10]'.format('x'.join(its_linear))])
        cmdline.extend(['--smoothing-sigmas', smooth_sig])
        cmdline.extend(['--shrink-factors', shrink_fac])

    if args.transform[0].rstrip('*') == 'Affine' or (args.transform[0].rstrip('*') == 'Syn' and not args.init_warp_dir_suffix):

        cmdline.extend(['--transform', 'Affine[0.1]'])

        w_img, w_lab = 1.0, 0.0
        if args.use_labels is not None:
            if len(args.use_labels) > weight_idx:
                w_lab = float(args.use_labels[weight_idx])
            w_img = 1.0 - w_lab
            weight_idx += 1

        if not args.template_mask:
            cmdline.extend(['--metric', 'MI[{},{},{},32,Regular,0.25]'.format(args.template_file[0], img_path, w_img)])
        else:
            cmdline.extend(['--metric', 'GC[{},{},{}]'.format(args.template_file[0], img_path, w_img)])

        if w_lab > 0.0:
            cmdline.extend(['--metric', 'MeanSquares[{},{},{}]'.format(args.use_labels[2], lab_path, w_lab)])

        cmdline.extend(['--convergence', '[{},1e-8,10]'.format('x'.join(its_linear))])
        cmdline.extend(['--smoothing-sigmas', smooth_sig])
        cmdline.extend(['--shrink-factors', shrink_fac])

    if args.transform[0].rstrip('*') == 'Syn':

        cmdline.extend(['--transform', 'SyN[0.1,3,0]'])

        w_img, w_lab = 1.0, 0.0
        if args.use_labels is not None:
            if len(args.use_labels) > weight_idx:
                w_lab = float(args.use_labels[weight_idx])
            w_img = 1.0 - w_lab
            weight_idx += 1

        cmdline.extend(['--metric', 'CC[{},{},{},4]'.format(args.template_file[0], img_path, w_img)])
        if w_lab > 0.0:
            cmdline.extend(['--metric', 'MeanSquares[{},{},{}]'.format(args.use_labels[2], lab_path, w_lab)])

        cmdline.extend(['--convergence', '[{},1e-9,15]'.format('x'.join(its_syn))])
        cmdline.extend(['--smoothing-sigmas', smooth_sig])
        cmdline.extend(['--shrink-factors', shrink_fac])

    #
    # mask

    if args.template_mask is not None:
        cmdline.extend(['--masks', args.template_mask[0]])

    #
    # launch

    print "Launching registration of file {}".format(img_file)

    name_list.append(img_file.split(os.extsep, 1)[0])
    launcher.add(name_list[-1], ' '.join(cmdline), args.out_dir[0])
    launcher.run(name_list[-1])

print "Waiting for registration jobs to finish..."

launcher.wait()

print "Registration finished."

