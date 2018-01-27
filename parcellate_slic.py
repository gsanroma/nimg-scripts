__author__ = 'gsanroma'

import nibabel as nib
import numpy as np
import argparse
import os
from skimage.segmentation import slic

parser = argparse.ArgumentParser()
parser.add_argument("segments",type=int)
parser.add_argument("compactness",type=int)
parser.add_argument("data_dir",type=str)
parser.add_argument("template_img",type=str)
parser.add_argument("template_mask",type=str)
parser.add_argument("parcel_file",type=str)

# args = parser.parse_args()
# args = parser.parse_args(['29000','100','/Users/gsanroma/DATA/stacking/NeoMater/parcellations','mask_d5.nii.gz','mask_d5.nii.gz','/Users/gsanroma/DATA/stacking/NeoMater/parcellations/parcel.nii.gz'])
args = parser.parse_args('5000 '
                         '100 '
                         '/home/sanromag/DATA/WMH/template/MNI_parcels/ '
                         'MNI152_boundary_mask.nii.gz '
                         'MNI152_boundary_mask.nii.gz '
                         '/home/sanromag/DATA/WMH/template/MNI_parcels/parcels.nii.gz '.split())


template_img = os.path.join(args.data_dir, args.template_img)
template_mask = os.path.join(args.data_dir, args.template_mask)

template_nib = nib.load(template_img)
template = template_nib.get_data().astype(np.float64)

mask_nib = nib.load(template_mask)
mask = mask_nib.get_data().astype(np.bool)

if args.segments > 1:

    template[~mask] = 1e30#np.finfo(np.float64).max/2.#np.inf#

    r = np.any(mask, axis=(1, 2))
    c = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))

    min0, max0 = np.where(r)[0][[0, -1]]
    min1, max1 = np.where(c)[0][[0, -1]]
    min2, max2 = np.where(z)[0][[0, -1]]

    template_crop = template[min0:max0, min1:max1, min2:max2]
    parcel_crop = slic(template_crop, n_segments=args.segments, compactness=args.compactness, multichannel=False)#, spacing=template_sitk.GetSpacing()[::-1])

    parcel = np.zeros(template.shape, dtype=np.int64)
    parcel[min0:max0, min1:max1, min2:max2] = parcel_crop
    parcel[~mask] = 0

else:

    parcel = np.zeros(template.shape, dtype=np.int64)
    parcel[mask] = 1

print "Number of parcels {}".format(np.unique(parcel).size - 1)

aux_nib = nib.Nifti1Image(parcel.astype(np.int32), template_nib.affine)#, template_nib.header)
nib.save(aux_nib, args.parcel_file)

pass