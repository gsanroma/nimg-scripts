import argparse
import os
import sys
from shutil import rmtree, copyfile
import numpy as np
import nibabel as nib
from scipy.spatial.distance import correlation, dice
from pickle import dump
from joblib import Parallel, delayed

parser = argparse.ArgumentParser('Computes the similarity between images according to some metric')
parser.add_argument("--in_dir", type=str, nargs=1, required=True)
parser.add_argument("--in_suffix", type=str, nargs=1, required=True)
parser.add_argument("--in2_dir", type=str, nargs=1, help="(optional) if not given, LOO in in_dir is used")
parser.add_argument("--in2_suffix", type=str, nargs=1, help="(optional) if not given, LOO in in_dir is used")
parser.add_argument("--mask_file", type=str, nargs=1, help="(optional) mask of region to compare")
parser.add_argument("--method", type=str, nargs='+', required=True, help='[Dice, [labels_list ...| nothing for all]] | Correlation | [NormalizedCorrelation, tmp_dir]')
parser.add_argument("--out_file", type=str, nargs=1, required=True, help="output file with pairwise similarities")
parser.add_argument("--num_procs", type=int, nargs=1, default=[8], help='number of concurrent processes ')
parser.add_argument("--num_itk_threads", type=int, nargs=1, default=[1], help='number of threads per proc ')

args = parser.parse_args()
# args = parser.parse_args('--in_dir /home/sanromag/DATA/OB/transformations/transforms_kk/ '
#                          '--in_suffix _OBV_S3Xtpl_Warped.nii.gz '
#                          '--mask_file /home/sanromag/DATA/OB/templates/mni152/mni152_A2Xtpl_OBVmask1.nii.gz '
#                          '--method Dice '
#                          '--out_file /home/sanromag/DATA/OB/templates/similarities/kk.dat '
#                          '--num_procs 5 '
#                          ''.split())

def avg_dice_distance(t1, t2, label_ids=None):

    if label_ids is None:
        ulab = np.unique(np.concatenate((np.unique(t1), np.unique(t2)), axis=0))
        ulab = np.delete(ulab, np.where(ulab==0))
    else:
        ulab = np.array(label_ids)

    count = 0.
    for i_lab in ulab:
        count += dice(t1 == i_lab, t2 == i_lab)

    retval = 0.
    if ulab.size > 0:
        retval = count / float(ulab.size)

    return retval


# is command line method ?
method_cmdline = False
if args.method[0] in ['NormalizedCorrelation']:
    method_cmdline = True

# if command-line method, then initialize launcher
launcher = None
check_file_repeat = None
if method_cmdline:
    # sys.path.insert(0, os.path.join(os.environ['HOME'], 'CODE', 'src', 'modules'))
    from scheduler import Launcher, check_file_repeat

    launcher = Launcher(args.num_procs[0])


# get file list

# sys.path.insert(0, os.path.join(os.environ['HOME'], 'CODE', 'src', 'modules'))
from utils import get_files_superlist

in1_names_list, files_superlist, _ = get_files_superlist(args.in_dir, args.in_suffix)
in1_files_list = files_superlist[0]
in1_dir = args.in_dir[0]

# get file list 2
if args.in2_dir is not None:
    in2_names_list, files_superlist = get_files_superlist(args.in2_dir, args.in2_suffix)
    in2_files_list = files_superlist[0]
    in2_dir = args.in2_dir[0]
else:
    in2_names_list = in1_names_list
    in2_files_list = in1_files_list
    in2_dir = in1_dir

# if command line method create temp dir and copy mask file
tmp_dir = None
if method_cmdline:
    tmp_dir = args.method[1]
    if os.path.exists(tmp_dir):
        rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    if args.mask_file is not None:
        copyfile(args.mask_file[0], os.path.join(tmp_dir, os.path.basename(args.mask_file[0])))

# create mask
mask = None
if args.mask_file is not None:
    mask_nib = nib.load(args.mask_file[0])
    mask = mask_nib.get_data()


scores = np.zeros((len(in1_files_list), len(in2_files_list)), dtype=np.float32)
for i1, (file1, name1) in enumerate(zip(in1_files_list, in1_names_list)):

    print "Computing similarities for file %s (%d out of %d)" % (name1, i1+1, len(in1_files_list))

    name_list = []

    if not method_cmdline:

        img1_nib = nib.load(os.path.join(in1_dir, file1))
        img1 = img1_nib.get_data()

        if args.mask_file is not None:
            assert all(np.equal(img1.shape, mask.shape)), "Target and mask images should be of same shape"
        else:
            mask = np.ones(img1.shape, dtype=np.bool)

        def sim(img1, file_path, method, mask):
            """Computes similarity between two images

            Parameters
            ----------
            img1 : numpy array
                image
            file_path: string
                path to the second image
            method: list
                list with method id and further parameters, if applicable

            Returns
            -------
            Similarity measure

            """
            img2_nib = nib.load(file_path)
            img2 = img2_nib.get_data()

            assert all(np.equal(img1.shape, img2.shape)), "Target2 and target2 should be of same shape"

            a = img1[mask.astype(np.bool)].ravel()
            b = img2[mask.astype(np.bool)].ravel()

            score = 0.

            if method[0] == 'Correlation':
                score = 1. - correlation(a, b)

            elif method[0] == 'Dice':

                label_ids = None
                if len(method) > 1:
                    label_ids = [int(method[i]) for i in range(1, len(method))]

                score = 1. - avg_dice_distance(a, b, label_ids)

            # print score

            return score

        scores_i1 = Parallel(n_jobs=args.num_procs[0])(delayed(sim)(img1, os.path.join(in2_dir, file2), args.method, mask) for file2 in in2_files_list)
        # scores_i1 = [sim(img1, os.path.join(in2_dir, file2), args.method) for file2 in in2_files_list]

        scores[i1] = scores_i1

    else:

        for i2, (file2, name2) in enumerate(zip(in2_files_list, in2_names_list)):

            cmdline = ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d' % args.num_itk_threads[0]]
            if args.method[0] == 'NormalizedCorrelation':
                imagemath_path = os.path.join(os.environ['ANTSPATH'], 'ImageMath')
                cmdline += [imagemath_path, '3', os.path.join(tmp_dir, 'dummy.txt'), 'NormalizedCorrelation']
                cmdline += [os.path.join(in1_dir, file1), os.path.join(in2_dir, file2)]
                if args.mask_file is not None:
                    cmdline += [os.path.join(tmp_dir, os.path.basename(args.mask_file[0]))]

            name_list.append('%sX%s' % (name1, name2))
            launcher.add(name_list[-1], ' '.join(cmdline), tmp_dir)
            launcher.run(name_list[-1])


    # Read scores when jobs are finished

    if method_cmdline:

        launcher.wait()

        for i2, name in enumerate(name_list):

            out_file = os.path.join(tmp_dir, name + '.out')
            check_file_repeat(out_file)

            try:
                with open(out_file) as f:
                    scores[i1, i2] = float(f.read().lstrip('-'))
            except:
                print('Error reading similarity value for file %s' % out_file)
                scores[i1, i2] = np.nan

            err_file = os.path.join(tmp_dir, name + '.err')
            sh_file = os.path.join(tmp_dir, name + '.sh')
            try:
                os.remove(out_file)
                os.remove(err_file)
                os.remove(sh_file)
            except:
                pass


with open(args.out_file[0], 'wb') as f:
    dump((in1_dir, in1_files_list, in2_dir, in2_files_list, scores), f)

if method_cmdline:
    rmtree(args.method[1])






