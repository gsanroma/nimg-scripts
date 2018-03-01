import argparse
import os
import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import binary_dilation

parser = argparse.ArgumentParser('Creates multiple masks (one for each file) or a single consensus mask for all files (if template is provided)')
parser.add_argument("--in_dir", type=str, nargs=1, required=True, help="dir of images or labelmaps")
parser.add_argument("--in_suffix", type=str, nargs=1, required=True, help="suffix for images or labelmaps")
parser.add_argument("--template_file", type=str, nargs=1, help="(optional) template file in case a single mask with all files")
# parser.add_argument("--threshold", type=str, nargs=1, default=[0.0], help="(optional) threshold in case of multiple masks")
parser.add_argument("--out_dir", type=str, nargs=1, help="dir of output mask (in case of multiple masks). Default: same as in_dir")
parser.add_argument("--out_suffix_or_mask_file", type=str, nargs=1, required=True, help="output suffix (multiple masks) or mask file (single mask)")
parser.add_argument("--dilation_radius", type=str, nargs=1, default=['1x1x1'], help="(optional) dilation radius (eg, 1x1x1)")

args = parser.parse_args()
# args = parser.parse_args('--in_dir /Users/gsanroma/DATA/DATABASES/STACOM17/templates '
#                          '--in_suffix _mask2.nii.gz '
#                          '--out_dir /Users/gsanroma/DATA/DATABASES/STACOM17/templates/ '
#                          '--out_suffix_or_mask_file _mask2.nii.gz '
#                          '--dilation_radius 5x5x5 '.split())


files_list = os.listdir(args.in_dir[0])
files_list = [f for f in files_list if f.endswith(args.in_suffix[0])]
assert files_list, "List of target labelmaps is empty"

single_mask = False
if args.template_file is not None:
    single_mask = True
    template_nib = nib.load(args.template_file[0])
    template = template_nib.get_data()
    img_size = template.shape
    mask = np.zeros(img_size, dtype=np.bool)

dilat_rad = [int(f) for f in args.dilation_radius[0].split('x')]
assert len(dilat_rad) == 3, 'Wrong dilation radius format'
struct = np.ones(dilat_rad, dtype=np.bool)

# output directory
out_dir = args.in_dir[0]
if args.out_dir is not None:
    out_dir = args.out_dir[0]
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

aux_nib = None

for file in files_list:

    print 'reading {}'.format(file)

    img_nib = nib.load(os.path.join(args.in_dir[0], file))
    img = img_nib.get_data()

    if single_mask:
        mask[img > 0] = True
        aux_nib = img_nib
    else:
        mask = np.zeros(img.shape, dtype=np.bool)
        mask[img > 0] = True
        mask = binary_dilation(mask, struct)
        mask_nib = nib.Nifti1Image(mask, img_nib.affine, img_nib.header)
        nib.save(mask_nib, os.path.join(out_dir, file.split(args.in_suffix[0])[0] + args.out_suffix_or_mask_file[0]))
        del mask

if single_mask:
    mask = binary_dilation(mask, struct)

    mask_nib = nib.Nifti1Image(mask, aux_nib.affine, aux_nib.header)
    nib.save(mask_nib, os.path.join(out_dir, args.out_suffix_or_mask_file[0]))