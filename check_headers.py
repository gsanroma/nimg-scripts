import argparse
import sys, os
import numpy as np
import nibabel as nib


parser = argparse.ArgumentParser(description='Checks the dimensions and orientations are consistent accross modalities of same subject')
parser.add_argument("--in_dir", type=str, nargs=1, required=True, help="Input directory containing the images")
parser.add_argument("--in_suffix_list", type=str, nargs='+', required=True, help="List of suffixes of different modalities")

args = parser.parse_args()
# args = parser.parse_args(''
#                          '--in_dir /home/sanromag/DATA/OB/data/data_t2_orig '
#                          '--in_suffix_list _t2.nii.gz _OBV.nii.gz _mask.nii.gz '
#                          ''.split())

assert len(args.in_suffix_list) > 1, 'must include more than one suffix'

# List of files

sys.path.insert(0, os.path.join(os.environ['HOME'], 'CODE', 'src', 'modules'))
from utils import get_files_superlist

names_list, files_superlist = get_files_superlist(args.in_dir, args.in_suffix_list)

#
# Read actual files

for i, files_list in enumerate(zip(*files_superlist)):

    print('Reading %s (%d of %d)' % (names_list[i], i + 1, len(names_list)))

    img_shape, img_affine = None, None

    for file in files_list:
        img = nib.load(os.path.join(args.in_dir[0], file))

        if img_shape is not None:
            if not np.allclose(img_shape, img.header.get_data_shape()):
                print('*** Shapes not equal!! ***')
            if not np.allclose(img_affine, img.affine.ravel(), atol=1e-3):
                print('*** Orientations not equal!! ***')

        else:
            img_shape = img.header.get_data_shape()
            img_affine = img.affine.ravel()



