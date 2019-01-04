import argparse
import os
import nibabel as nib
import numpy as np
from shutil import copyfile
import sys

parser = argparse.ArgumentParser(description='Processes the images including N4 correction and histogram matching to template.\n'
                                             'Optionally, images and/or template can be masked out (e.g., remove skull) given the mask file.')
parser.add_argument("--in_dir", type=str, nargs=1, required=True, help='directory input images')
parser.add_argument("--img_suffix_list",type=str, nargs='+', required=True, help='list of suffixes of input images')
parser.add_argument("--mask_suffix_list", type=str, nargs='+', help="list of mask suffixes for processing (same for all modalities if only one given)")
parser.add_argument("--label_suffix", type=str, nargs=1, help="transfer label files to the output directory")
parser.add_argument("--maskout_img", action="store_true", help="Maskout image (--mask_suffix is required).")
parser.add_argument("--maskout_lab", action="store_true", help="Maskout label (--mask_suffix is required). Labels are masked out with 1st mask in the list")
parser.add_argument("--denoising", type=int, nargs=1, default=[0], help="Denoising shrink factor (if 0 then not done. Default 0)")
parser.add_argument("--n4", type=int, nargs=1, default=[0], help="N4 bias correction shrink factor (if 0 then not done. Default 0)")
parser.add_argument("--histmatch", action="store_true", help="histogram matching (needs template)")
parser.add_argument("--normalize", action="store_true", help="subtract mean and divide std (in mask region if mask is provided)")
parser.add_argument("--template_file", type=str, nargs=1, help="template image")
parser.add_argument("--template_maskout_mask", type=str, nargs=1, help="maskout template")
parser.add_argument("--template_norm", action="store_true", help="normalize template intensities between [0..1]")
parser.add_argument("--out_dir", type=str, nargs=1, required=True, help='directory to store processed imges')
parser.add_argument("--num_procs", type=int, nargs=1, default=[8], help='number of concurrent processes ')
parser.add_argument("--num_itk_threads", type=int, nargs=1, default=[1], help='number of threads per ANTs proc ')


args = parser.parse_args()
# args = parser.parse_args('--in_dir /home/sanromag/DATA/WMH/train_RS/data_denoise2/ --img_suffix_list _FLAIR.nii.gz --maskout_suffix _brainmaskWarped.nii.gz --normalize --out_dir /home/sanromag/DATA/WMH/train_RS/data_denoise_norm/'.split())

# start launcher and specify max amount of processes
sys.path.insert(0, os.path.join(os.environ['HOME'], 'CODE', 'modules'))
from scheduler import Launcher

launcher = Launcher(args.num_procs[0])

#
# Initial checks
#

if not args.histmatch:
    assert args.template_file is None and args.template_maskout_mask is None and not args.template_norm, "Unnecessary template if not histmatch"

if args.histmatch:
    assert args.template_file is not None, "Need template for histogram matching"

if args.template_file is not None:
    assert os.path.exists(args.template_file[0]), "Template file not found"

if args.template_maskout_mask is not None:
    assert os.path.exists(args.template_maskout_mask[0]), "Template mask not found"

files_list = os.listdir(args.in_dir[0])
name_list = [f.split(args.img_suffix_list[0])[0] for f in files_list if f.endswith(args.img_suffix_list[0])]
img_superlist = [[f + suffix for f in name_list] for suffix in args.img_suffix_list]
# print("%s" % img_superlist)
# a = [os.path.exists(os.path.join(args.in_dir[0], f)) for img_list in img_superlist for f in img_list]
# print("%s" % a)
assert False not in [os.path.exists(os.path.join(args.in_dir[0], f)) for img_list in img_superlist for f in img_list], 'Some image not found'

# create output directory
if not os.path.exists(args.out_dir[0]):
    os.makedirs(args.out_dir[0])

# create mask file list and copy
mask_superlist = [[]]
if args.mask_suffix_list is not None:
    assert len(args.mask_suffix_list) == 1 or len(args.mask_suffix_list) == len(args.img_suffix_list), 'Num of mask suffixes must be 1 or equal to num img suffixes'
    # mask_list = [f + args.mask_suffix[0] for f in name_list]
    mask_superlist = [[f + suffix for f in name_list] for suffix in args.mask_suffix_list]
    assert False not in [os.path.exists(os.path.join(args.in_dir[0], f)) for mask_list in mask_superlist for f in mask_list]
    # copy masks
    for mask_list in mask_superlist:
        for mask_file in mask_list:
            copyfile(os.path.join(args.in_dir[0], mask_file), os.path.join(args.out_dir[0], mask_file))

# make mask superlist equal length as img superlist
if len(mask_superlist) == 1:
    mask_superlist *= len(img_superlist)

# create labels file list and copy
label_list = []
if args.label_suffix is not None:
    label_list = [f + args.label_suffix[0] for f in name_list]
    assert False not in [os.path.exists(os.path.join(args.in_dir[0], f)) for f in label_list]
    # copy labels
    for lab_file in label_list:
        copyfile(os.path.join(args.in_dir[0], lab_file), os.path.join(args.out_dir[0], lab_file))

# input directory
in_dir = args.in_dir[0]

#
# Pipeline
#

if args.maskout_img or args.maskout_lab:

    assert args.mask_suffix_list is not None, "Mask suffix is required for maskout"

    if args.maskout_img:

        for img_list, mask_list in zip(img_superlist, mask_superlist):

            for img_file, mask_file in zip(img_list, mask_list):

                print("Masking out image %s" % img_file)

                # mask
                mask_nib = nib.load(os.path.join(in_dir, mask_file))
                mask = mask_nib.get_data().astype(np.bool)

                # image
                img_nib = nib.load(os.path.join(args.in_dir[0], img_file))
                img = img_nib.get_data()
                img[~mask] = img.min()
                aux = nib.Nifti1Image(img, img_nib.affine, img_nib.header)
                nib.save(aux, os.path.join(args.out_dir[0], img_file))

    if args.maskout_lab:

        assert args.label_suffix is not None, 'Label suffix is required for masking out labels'

        for lab_file, mask_file in zip(label_list, mask_superlist[0]):

            print("Masking out labels %s" % lab_file)

            # mask
            mask_nib = nib.load(os.path.join(in_dir, mask_file))
            mask = mask_nib.get_data().astype(np.bool)

            # label
            lab_nib = nib.load(os.path.join(args.in_dir[0], lab_file))
            lab = lab_nib.get_data()
            lab[~mask] = lab.min()
            aux = nib.Nifti1Image(lab, lab_nib.affine, lab_nib.header)
            nib.save(aux, os.path.join(args.out_dir[0], lab_file))

    # assign input directory for subsequent steps
    in_dir = args.out_dir[0]


if args.denoising[0] != 0:

    denoise_path = os.path.join(os.environ['ANTSPATH'], 'DenoiseImage')

    name_list = []

    for img_list, mask_list in zip(img_superlist, mask_superlist):

        for img_file, mask_file in map(None, img_list, mask_list):

            cmdline = ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d' % args.num_itk_threads[0], denoise_path, '--image-dimensionality', '3']
            cmdline.extend(['--input-image', os.path.join(in_dir, img_file)])
            # cmdline.extend(['--noise-model', 'Rician'])
            cmdline.extend(['--shrink-factor', '%d' % args.denoising[0]])
            # cmdline.extend(['--verbose', '1'])
            if mask_file is not None:
                cmdline.extend(['--mask-image', os.path.join(in_dir, mask_file)])
            cmdline.extend(['--output', os.path.join(args.out_dir[0], img_file)])

            name_list.append(img_file.split(os.extsep, 1)[0])
            launcher.add(name_list[-1], ' '.join(cmdline), args.out_dir[0])
            launcher.run(name_list[-1])

            print("Launched Denoising for %s" % img_file)

    print("Waiting for denoising...")
    launcher.wait()

    print("Denoising finished.")

    # assign input directory for subsequent steps
    in_dir = args.out_dir[0]


if args.n4[0] != 0:

    n4_path = os.path.join(os.environ['ANTSPATH'], 'N4BiasFieldCorrection')

    name_list = []

    for img_list, mask_list in zip(img_superlist, mask_superlist):

        for img_file, mask_file in map(None, img_list, mask_list):

            cmdline = ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d' % args.num_itk_threads[0], n4_path, '--image-dimensionality', '3']
            cmdline.extend(['--input-image', os.path.join(in_dir, img_file)])
            cmdline.extend(['--shrink-factor', '%d' % args.n4[0]])
            if mask_file is not None:
                cmdline.extend(['--mask-image', os.path.join(in_dir, mask_file)])
            cmdline.extend(['--convergence', '50x50x30x20', '1e-6'])
            cmdline.extend(['--bspline-fitting', '300'])
            cmdline.extend(['--output', os.path.join(args.out_dir[0], img_file)])

            name_list.append(img_file.split(os.extsep, 1)[0])
            launcher.add(name_list[-1], ' '.join(cmdline), args.out_dir[0])
            launcher.run(name_list[-1])

            print("Launched N4 for %s" % img_file)

    print("Waiting for N4...")
    launcher.wait()

    print("N4 finished.")

    # assign input directory for subsequent steps
    in_dir = args.out_dir[0]


if args.histmatch:

    #
    # process template

    if args.template_maskout_mask is not None or args.template_norm:

        print("Processing template")

        template_nib = nib.load(args.template_file[0])
        template = template_nib.get_data()

        # mask out template file
        if args.template_maskout_mask is not None:
            mask_nib = nib.load(args.template_maskout_mask[0])
            mask = mask_nib.get_data().astype(np.bool)
            template[~mask] = template.min()
            copyfile(args.template_maskout_mask[0], os.path.join(args.out_dir[0], os.path.basename(args.template_maskout_mask[0])))

        if args.template_norm:
            template = (template - template.min()) / (template.max() - template.min())

        aux = nib.Nifti1Image(template, template_nib.affine, template_nib.header)
        nib.save(aux, os.path.join(args.out_dir[0], os.path.basename(args.template_file[0])))

    else:
        copyfile(args.template_file[0], os.path.join(args.out_dir[0], os.path.basename(args.template_file[0])))

    imagemath_path = os.path.join(os.environ['ANTSPATH'],'ImageMath')

    name_list = []

    for img_list in img_superlist:

        for img_file in img_list:

            in_file = os.path.join(in_dir, img_file)
            out_file = os.path.join(args.out_dir[0], img_file)
            tpl_file = os.path.join(args.out_dir[0], os.path.basename(args.template_file[0]))

            cmdline = [imagemath_path, '3', out_file, 'HistogramMatch', in_file, tpl_file]

            name_list.append(img_file.split(os.extsep, 1)[0])
            launcher.add(name_list[-1], ' '.join(cmdline), args.out_dir[0])
            launcher.run(name_list[-1])

            print("Launched histogram match of %s" % img_file)

    print("Waiting for Histogram matching...")
    launcher.wait()

    print("Histogram matching finished.")

    # assign input directory for subsequent steps
    in_dir = args.out_dir[0]

if args.normalize:

    from scipy.stats.mstats import zscore

    for img_list, mask_list in zip(img_superlist, mask_superlist):

        for img_file, mask_file in map(None, img_list, mask_list):

            print("Normalizing subject %s" % img_file)

            # image
            img_nib = nib.load(os.path.join(in_dir, img_file))
            img = img_nib.get_data()

            # mask
            mask = np.ones(img.shape, np.bool)
            if mask_file is not None:
                mask_nib = nib.load(os.path.join(in_dir, mask_file))
                mask = mask_nib.get_data().astype(np.bool)

            # normalize
            img2 = np.zeros(img.shape, dtype=np.float32)
            img2[mask] = zscore(img[mask].astype(np.float32))
            aux = nib.Nifti1Image(img2, img_nib.affine, img_nib.header.set_data_dtype(np.float32))
            nib.save(aux, os.path.join(args.out_dir[0], img_file))




