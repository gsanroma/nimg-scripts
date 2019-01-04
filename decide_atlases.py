__author__ = 'gsanroma'

import argparse
import numpy as np
import os
import sys
from sklearn import manifold
from sklearn import cluster
from scipy.spatial import distance
import matplotlib


# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt


#
# # Decides images at each fold by maximizing the spread among the selected images in the manifold
#


parser = argparse.ArgumentParser('Selects one (or multiple) sets of images evenly spread accross the population according to some similarity metric.'
                                 'Useful to select train (atlases) and test (target) sets for cross-validation.')
parser.add_argument("--scores_file", type=str, nargs=1, required=True)
parser.add_argument("--use_subset", type=str, nargs=3, help='(optional) img_dir, img_suffix, scores_img_suffix. Use subset of files')
parser.add_argument("--num_folds", type=int, nargs=1, help="(optional) number of folds")
parser.add_argument("--num_atlas", type=int, nargs=1, help="(optional) number of atlases (assumes single fold)")
parser.add_argument("--num_val", type=int, nargs=1, default=[0], help="(optional) number of validation samples within the atlas set")
parser.add_argument("--out_fig", type=str, nargs=1)
parser.add_argument("--create_symlinks", type=str, nargs=5, help='symlink_dir_prefix, origin_dir, img_suffix, label_suffix, scores_img_suffix')

args = parser.parse_args()
# args = parser.parse_args('--scores_file /home/sanromag/DATA/OB/templates/Dice_S3mXtpl.dat --use_subset /home/sanromag/DATA/OB/data_orig _t2.nii.gz _OBV_S3mXtpl_Warped.nii.gz --num_atlas 50 --out_fig /home/sanromag/DATA/OB/atlases.png'.split())
# args = parser.parse_args('--scores_file /home/sanromag/DATA/OB/templates/Dice_S3mXtpl.dat --num_atlas 50 --out_fig /home/sanromag/DATA/OB/atlases.png'.split())

sys.path.insert(0, os.path.join(os.environ['HOME'], 'CODE', 'modules'))
from utils import read_sim_scores, get_files_superlist

scores_dict = read_sim_scores(args.scores_file[0])

assert scores_dict['in1_dir'] == scores_dict['in2_dir'] and scores_dict['in1_files_list'] == scores_dict['in2_files_list'], "Not same files"
assert scores_dict['scores'].shape[0] == scores_dict['scores'].shape[1], "scores file is not square"

# use a subset of the scores
Idx = range(len(scores_dict['in1_files_list']))  # by default use all files
if args.use_subset is not None:
    # read the file list
    img_names_list, files_superlist = get_files_superlist([args.use_subset[0]], [args.use_subset[1]])
    img_files_list = files_superlist[0]
    img_dir = args.use_subset[0]
    # get the corresponding indices in scores file
    Idx = []
    for img_name in img_names_list:
        found = False
        for i in range(len(scores_dict['in1_files_list'])):
            if img_name == scores_dict['in1_files_list'][i].split(args.use_subset[2])[0]:
                Idx.append(i)
                found = True
                break
        assert found, 'Filename %s in subset not found in scores file' % img_name

# create variables containing scores matrix and files list
scores = scores_dict['scores'][np.ix_(Idx, Idx)]
in_files_list = [scores_dict['in1_files_list'][i] for i in Idx]

Nimgs = scores.shape[0]

Ntargets = []
n_folds = 1
if args.num_folds is not None:
    n_folds = args.num_folds[0]
    q = Nimgs // n_folds
    r = Nimgs % n_folds
    Ntargets = [q] * n_folds
    for i in range(r): Ntargets[i] += 1
else:
    Natlas = args.num_atlas[0]


#
# Select the atlases most spread in the manifold
#

Y = manifold.SpectralEmbedding(n_components=2, affinity='precomputed').fit_transform(scores)

atlas_idx = []
target_idx = []
val_idx = []

if args.num_folds is not None:

    indexes = np.arange(Nimgs)

    for i_fold in range(n_folds):

        if i_fold < n_folds - 1:

            labels = cluster.KMeans(Ntargets[i_fold], random_state=1234).fit_predict(Y[indexes])

            # Computation of cluster representatives
            centers = np.zeros((Ntargets[i_fold], Y.shape[1]))
            for i in range(Ntargets[i_fold]):
                centers[i] = np.mean(Y[indexes[labels == i]], axis=0)

            D = distance.cdist(centers, Y[indexes], metric='euclidean', p=2)

            target_idx += [list(indexes[list(np.argmin(D, axis=1))])]
            remaining_idx = np.array(list(set(range(Nimgs)) - set(target_idx[-1])))

        else:

            assert indexes.size == Ntargets[-1], "number of remaining indices do not coincide with number of targets in last fold"
            target_idx += [list(indexes)]
            remaining_idx = np.array(list(set(range(Nimgs)) - set(target_idx[-1])))

        #
        # validation set

        if args.num_val[0] > 0:

            labels = cluster.KMeans(args.num_val[0], random_state=1234).fit_predict(Y[remaining_idx])

            # Computation of cluster representatives
            centers = np.zeros((args.num_val[0], Y.shape[1]))
            for i in range(args.num_val[0]):
                centers[i] = np.mean(Y[remaining_idx[labels == i]], axis=0)

            D = distance.cdist(centers, Y[remaining_idx], metric='euclidean', p=2)

            val_idx += [list(remaining_idx[list(np.argmin(D, axis=1))])]
            atlas_idx += [list(set(range(Nimgs)) - set(target_idx[-1]) - set(val_idx[-1]))]

        else:
            atlas_idx += [list(remaining_idx)]
            val_idx += [[]]

        #
        # Remove current targets from set of available indices

        indexes = np.array(list(set(indexes) - set(target_idx[-1])))

else:

    labels = cluster.KMeans(Natlas, random_state=1234).fit_predict(Y)
    centers = np.zeros((Natlas, Y.shape[1]))
    for i in range(Natlas): centers[i] = np.mean(Y[labels == i], axis=0)
    D = distance.cdist(centers, Y, metric='euclidean', p=2)
    remaining_idx = np.array(list(np.argmin(D, axis=1)))
    target_idx = [list(set(range(Nimgs)) - set(list(remaining_idx)))]
    # validation set
    if args.num_val[0] > 0:
        labels = cluster.KMeans(args.num_val[0], random_state=1234).fit_predict(Y[remaining_idx])
        centers = np.zeros((args.num_val[0], Y.shape[1]))
        for i in range(args.num_val[0]):
            centers[i] = np.mean(Y[remaining_idx[labels == i]], axis=0)
        D = distance.cdist(centers, Y[remaining_idx], metric='euclidean', p=2)
        val_idx += [list(remaining_idx[list(np.argmin(D, axis=1))])]
        atlas_idx += [list(set(remaining_idx) - set(val_idx[0]))]
    else:
        atlas_idx = [list(remaining_idx)]
        val_idx = [[]]


# for i in range(n_folds):
#     assert set(atlas_idx[i] + target_idx[i] + val_idx[i]) == set(range(Nimgs))
# print('All is OK')

#
# Plot the atlases in the manifold
#

if args.out_fig is not None:

    n_colors = 3 if args.num_val > 0 else 2

    distinct_colors = plt.get_cmap('jet')(np.linspace(0, 1.0, n_colors))

    plt.figure()

    for i_fold in range(n_folds):

        plt.subplot('1%d%d' % (n_folds, i_fold))
        plt.scatter(Y[:, 0], Y[:, 1], s=10, c=distinct_colors[0], cmap=plt.cm.Spectral)

        plt.title("Atlases in fold %d" % i_fold)

        for i_img in range(Nimgs):
            plt.annotate('%d' % i_img, (Y[i_img, 0], Y[i_img, 1]), fontsize=7)

        Yatlas = Y[atlas_idx[i_fold]]
        plt.scatter(Yatlas[:, 0], Yatlas[:, 1], s=60, facecolors='none', edgecolors=distinct_colors[1], cmap=plt.cm.Spectral)

        if args.num_val[0] > 0:
            Yval = Y[val_idx[i_fold]]
            plt.scatter(Yval[:, 0], Yval[:, 1], s=60, facecolors='none', edgecolors=distinct_colors[2], cmap=plt.cm.Spectral, linestyle='dashed')

        plt.savefig(args.out_fig[0])


#
# Create symlinks
#

if args.create_symlinks is not None:

    for i_fold in range(n_folds):

        if n_folds > 1:
            train_dir = args.create_symlinks[0] + '_train%d' % i_fold
            test_dir = args.create_symlinks[0] + '_test%d' % i_fold
            val_dir = args.create_symlinks[0] + '_val%d' % i_fold
        else:
            train_dir = args.create_symlinks[0] + '_train'
            test_dir = args.create_symlinks[0] + '_test'
            val_dir = args.create_symlinks[0] + '_val'

        os.makedirs(train_dir)
        os.makedirs(test_dir)
        os.makedirs(val_dir)

        for idx_list, out_dir in zip([atlas_idx[i_fold], target_idx[i_fold], val_idx[i_fold]], [train_dir, test_dir, val_dir]):
            for i in idx_list:
                img_file = in_files_list[i].split(args.create_symlinks[4])[0] + args.create_symlinks[2]
                lab_file = in_files_list[i].split(args.create_symlinks[4])[0] + args.create_symlinks[3]
                os.symlink(os.path.join(args.create_symlinks[1], img_file ), os.path.join(out_dir, img_file))
                os.symlink(os.path.join(args.create_symlinks[1], lab_file ), os.path.join(out_dir, lab_file))


