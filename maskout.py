import nibabel as nib
import numpy as np
import os
import sys
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser('Masks-out images given a mask')
parser.add_argument("--in_dir", type=str, nargs=1, required=True, help='input directory')
parser.add_argument("--in_suffix_list", type=str, nargs='+', required=True, help='list of suffixes of images to maskout')
parser.add_argument("--mask_suffix", type=str, nargs=1, help='suffix of mask files (in case each file with its own mask)')
parser.add_argument("--mask_file", type=str, nargs=1, help='mask file (in case all images with same mask, eg, as in template space)')
parser.add_argument("--maskout_inside", action='store_true', help='maskout inside mask (outside by default)')
parser.add_argument("--out_dir", type=str, nargs=1, help='(optional) output directory (if not spec., same as in_dir)')
parser.add_argument("--out_suffix_list", type=str, nargs='+', help='(optional) list of out suffixes (if not spec. same as in_suffix_list)')
parser.add_argument("--num_procs", type=int, nargs=1, default=[8], help='number of concurrent processes ')

args = parser.parse_args()
# args = parser.parse_args('--in_dir /home/sanromag/WORK/DATA/VBM_1158_tissue_WMH/tmp/kk --in_suffix_list _T1.nii.gz --mask_suffix _aseg.nii.gz --out_dir /home/sanromag/WORK/DATA/VBM_1158_tissue_WMH/tmp/kk --out_suffix_list _T1maskout.nii.gz'.split())

# assign output directory
out_dir = args.in_dir[0]
if args.out_dir is not None:
    out_dir = args.out_dir[0]

# assign output suffix list
out_suffix_list = args.in_suffix_list
if args.out_suffix_list is not None:
    assert len(args.in_suffix_list) == len(args.out_suffix_list), 'in and out suffix list must be of same size'
    out_suffix_list = args.out_suffix_list

# get files superlist
sys.path.insert(0, os.path.join(os.environ['HOME'], 'CODE', 'modules'))
from utils import get_files_superlist

# note that mask suffix is appended in the end
unique_mask = False
if args.mask_suffix is not None:
    names_list, files_superlist = get_files_superlist(args.in_dir, args.in_suffix_list + args.mask_suffix)
elif args.mask_file is not None:
    names_list, files_superlist = get_files_superlist(args.in_dir, args.in_suffix_list)
    unique_mask = True
else:
    assert False, 'either suffix list or suffix file must be given'


# open unique mask in case template space
mask = None
if unique_mask:
    mask = nib.load(args.mask_file[0]).get_data().astype(np.bool)
    if args.maskout_inside == False:
        mask = ~mask

def maskout(args, name, files_list, mask=None):
    """Maskout an image list

    Parameters
    ----------
    name: str
        name of the image
    files_list: list
        list of files to maskout (if mask is None, then last element is the mask)
    mask: numpy.array
        mask for maskout. If None then mask is taken from last element of files_list
    """

    print 'processing %s' % name

    # print files_list

    aux_files_list = files_list  # all files in list have to be masked out
    if mask is None:
        # mask file is in the end of the list
        mask = nib.load(os.path.join(args.in_dir[0], files_list[-1])).get_data().astype(np.bool)
        if args.maskout_inside == False:
            mask = ~mask
        aux_files_list = files_list[:-1]  # last file in list is mask, so we exclude it

    for file, suffix in zip(aux_files_list, out_suffix_list):  # maskout image files

        img_nib = nib.load(os.path.join(args.in_dir[0], file))
        img = img_nib.get_data()
        img[mask] = 0

        img2_nib = nib.Nifti1Image(img, img_nib.affine, img_nib.header)
        nib.save(img2_nib, os.path.join(out_dir, name + suffix))


# maskout and save

Parallel(n_jobs=args.num_procs[0])(delayed(maskout)(args, name, files_list, mask) for name, files_list in zip(names_list, zip(*files_superlist)))

# for name, files_list in zip(names_list, zip(*files_superlist)):
#     maskout(name, files_list, mask)




