
import os
import argparse
from pickle import load
import numpy as np
from subprocess import call
import sys
from shutil import rmtree
from copy import copy

from scheduler import Launcher

#
# PATHS
#

code_dir = os.path.join(os.environ['HOME'], 'CODE', 'src')
warptotemplate_path = os.path.join(code_dir, 'nimg-scripts', 'warp_to_template.py')
warpatlasestotarget_path = os.path.join(code_dir, 'nimg-scripts', 'warp_atlases_to_target.py')
maskout_path = os.path.join(code_dir, 'nimg-scripts', 'maskout.py')
imagemath_path = os.path.join(os.environ['ANTSPATH'], 'ImageMath')
# imagemath_path = os.path.join('home', 'sanromag', 'Programs', 'ANTs', 'build', 'bin', 'ImageMath')
pblf_path = os.path.join(code_dir, 'nimg-scripts', 'pblf.py')


#
# Label Fusion function
#

# # NEED TO WAIT FOR JOINING RESULTING LABELS (if join_labels==True) not possible to set it False from python script (only calling function)
# # In case joining labels, need to set job_id to the joining job

def label_fusion(launcher, target_path, atlas_img_path_list, atlas_lab_path_list, out_file, probabilities, method, metric,
                 target_mask_path=None, patch_rad=None, search_rad=None, fusion_rad=None, struct_sim=None, patch_norm=None,
                 params_list=None, joint_alpha=None, joint_beta=None, joint_metric=None, parallel_label_superlist=None,
                 num_itk_threads=None, target_lab_4_metrics=None):

    out_dir = os.path.dirname(out_file)
    out_name = os.path.basename(out_file).split(os.extsep, 1)[0]
    # target_tmp_dir = os.path.join(out_dir, out_name)

    if method == 'joint' or (method == 'majvot' and not probabilities):

        if method == 'joint':

            jointfusion_path = os.path.join(os.environ['ANTSPATH'], 'antsJointFusion')

            cmdline = ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d' % num_itk_threads, jointfusion_path]
            cmdline.extend(['-d', '3'])
            cmdline.extend(['-t', target_path])
            cmdline.extend(['-g'] + atlas_img_path_list)
            cmdline.extend(['-l'] + atlas_lab_path_list)

            if target_mask_path is not None:
                cmdline.extend(['-x', target_mask_path])

            # cmdline.extend(['-m', 'Joint'])
            if patch_rad is not None: cmdline.extend(['-p', patch_rad])
            if search_rad is not None: cmdline.extend(['-s', search_rad])

            if joint_alpha is not None: cmdline.extend(['-a', '%f' % joint_alpha])
            if joint_beta is not None: cmdline.extend(['-b', '%f' % joint_beta])
            if joint_metric is not None: cmdline.extend(['-m', joint_metric])

            if probabilities:
                prob_path = os.path.join(out_dir, out_name + '_prob%d.nii.gz')
                int_path = os.path.join(out_dir, out_name + '_int%d.nii.gz')
                cmdline.extend(['-o [%s,%s,%s]' % (out_file, int_path, prob_path)])
            else:
                cmdline.extend(['-o', out_file])

        elif method == 'majvot':

            cmdline = ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d' % num_itk_threads, imagemath_path, '3', out_file, 'MajorityVoting'] + atlas_lab_path_list

        # launch
        launcher.add(out_name, ' '.join(cmdline), out_dir)
        launcher.run(out_name)



    elif method == 'nlwv' or method == 'lasso' or method == 'nlbeta' or method == 'myjoint' or method == 'deeplf' or (method == 'majvot' and probabilities):

        cmdline = ['python', '-u', pblf_path]

        cmdline.extend(['--atlas_img_list'] + atlas_img_path_list)
        cmdline.extend(['--target_img', target_path])
        cmdline.extend(['--atlas_lab_list'] + atlas_lab_path_list)
        cmdline.extend(['--method', method])
        cmdline.extend(['--metric', metric])

        if method == 'nlwv' or method == 'lasso' or method == 'nlbeta' or method == 'myjoint' or method == 'deeplf':

            cmdline.extend(['--search_radius', search_rad])
            cmdline.extend(['--fusion_radius', fusion_rad])
            cmdline.extend(['--struct_sim', '%f' % struct_sim])

        if method == 'nlwv' or method == 'lasso' or method == 'nlbeta' or method == 'myjoint':

            cmdline.extend(['--patch_radius', patch_rad])
            cmdline.extend(['--normalization', patch_norm])

        if parallel_label_superlist is not None:  # TO BE UPDATED WITH ALL THE STUFF ADDED TO THE ELSE CASE

            #
            # Parallel execution

            # if not os.path.exists(target_tmp_dir): os.makedirs(target_tmp_dir)

            assert len(params_list) == 1 or len(params_list) == len(parallel_label_superlist)

            for i, label_list in enumerate(parallel_label_superlist):

                cmdline_aux = copy(cmdline)
                cmdline_aux.extend(['--label_grp'] + ['%d' % label_id for label_id in label_list])

                if method == 'nlwv' or method == 'lasso' or method == 'nlbeta' or method == 'myjoint':
                    cmdline_aux.extend(['--regularization', '%f' % (params_list[i] if len(params_list) > 1 else params_list[0])])
                elif method == 'deeplf':
                    cmdline_aux.extend(['--load_net', '%s' % (params_list[i] if len(params_list) > 1 else params_list[0])])

                if probabilities: cmdline_aux.extend(['--prob_suffix_pattern', '_prob%d.nii.gz'])

                if target_lab_4_metrics is not None: cmdline_aux.extend(['--classification_metrics', '%s' % target_lab_4_metrics])

                cmdline_aux.extend(['--out_file', os.path.join(out_dir, '%s_grp%d.nii.gz' % (out_name, i))])
                # cmdline_aux.extend(['--out_file', os.path.join(target_tmp_dir, 'label%d.nii.gz' % label_id)])

                # launch
                out_name_aux = '%s_g%d' % (out_name, i)
                launcher.add(out_name_aux, ' '.join(cmdline_aux), out_dir)
                launcher.run(out_name_aux)


        else:

            #
            # Normal execution

            # assert len(params_list) == 1

            if method == 'nlwv' or method == 'lasso' or method == 'nlbeta' or method == 'myjoint':
                cmdline.extend(['--regularization', '%f' % (params_list[0])])
            elif method == 'deeplf':
                cmdline.extend(['--load_net', '%s' % (params_list[0])])

            if probabilities: cmdline.extend(['--prob_suffix_pattern', '_prob%d.nii.gz'])

            if target_lab_4_metrics is not None: cmdline.extend(['--classification_metrics', '%s' % target_lab_4_metrics])

            cmdline.extend(['--out_file', out_file])

            # launch
            launcher.add(out_name, ' '.join(cmdline), out_dir)
            launcher.run(out_name)

    return launcher


def atlas_selection(args_atsel, target_names, atlas_names, is_loo=False):

    print('Reading scores for atlas selection')

    # load scores file
    f = open(args_atsel[1], 'r')
    in_dir, in_files_list, in2_dir, in2_files_list, scores_aux = load(f)
    f.close()

    s_target_names = [f.split(args_atsel[2])[0] for f in in_files_list]
    s_atlas_names = [f.split(args_atsel[3])[0] for f in in2_files_list]

    assert set(target_names).issubset(set(s_target_names)), "Image filenames are not subset of score filenames"
    if not is_loo:
        assert set(atlas_names).issubset(set(s_atlas_names)), "Atlas filenames are not subset of score filenames"

    Ntar, Natl = len(target_names), len(atlas_names)
    scores = np.zeros((Ntar, Natl), dtype=np.float32)

    for i_t in range(Ntar):
        i2_t = [i for i, f in enumerate(s_target_names) if target_names[i_t] == f][0]
        for i_a in range(Natl):
            if is_loo and target_names[i_t] == atlas_names[i_a]:
                continue
            i2_a = [i for i, f in enumerate(s_atlas_names) if atlas_names[i_a] == f][0]
            scores[i_t, i_a] = scores_aux[i2_t, i2_a]

    return np.argsort(scores, axis=1)[:, :-int(args_atsel[0]) - 1:-1]


def get_label_fusion_params(target_dir, target_img_suffix, target_mask_suffix=None, reg_dir=None, reg_img_suffix=None, reg_lab_suffix=None, args_atsel=None):

    files = os.listdir(target_dir)
    target_img_list = [f for f in files if f.endswith(target_img_suffix)]
    target_name_list = [f.split(target_img_suffix)[0] for f in target_img_list]
    target_path_list = [os.path.join(target_dir, f) for f in target_img_list]
    assert target_img_list, 'No target image found'

    target_mask_list = None
    if target_mask_suffix is not None:
        target_mask_list = [f.split(target_img_suffix)[0] + target_mask_suffix for f in target_img_list]

    atlas_dir_list = [os.path.join(reg_dir, f.split(target_img_suffix)[0]) for f in target_img_list]
    atlas_img_path_superlist, atlas_lab_path_superlist = [], []
    for i, atlas_dir in enumerate(atlas_dir_list):
        files = os.listdir(atlas_dir)
        atlas_img_list = [f for f in files if f.endswith(reg_img_suffix)]
        assert atlas_img_list, 'No atlas image found'
        atlas_name_list = [f.split(reg_img_suffix)[0] for f in atlas_img_list]
        atlas_idx = np.array(range(len(atlas_name_list)))
        if args_atsel is not None:
            atlas_idx = atlas_selection(args_atsel, [target_name_list[i]], atlas_name_list)[0]
        # print('atlas dir: %s, atlas idx %s' % (atlas_dir, atlas_idx))
        # print('atlas_name_list: %s' % atlas_name_list)
        atlas_img_path_list = [os.path.join(atlas_dir, atlas_img_list[j]) for j in atlas_idx]
        atlas_lab_path_list = [os.path.join(atlas_dir, atlas_name_list[j]) + reg_lab_suffix for j in atlas_idx]


        #
        #
        # assert False not in [os.path.exists(f) for f in atlas_lab_path_list]
        # THIS IS TO AVOID THE SCRIPT STOPPING BECAUSE ONE STUPID LABEL FILE DID NOT WARP, but should be checked what happened
        idx_list = [j for j, f in enumerate(atlas_lab_path_list) if not os.path.exists(f)]
        for idx in sorted(idx_list, reverse=True):
            print '****** LABEL FOR %s DOES NOT EXIST ****** ' % atlas_img_path_list[idx]
            del atlas_img_path_list[idx]
            del atlas_lab_path_list[idx]
        #
        #
        #


        atlas_img_path_superlist.append(atlas_img_path_list)
        atlas_lab_path_superlist.append(atlas_lab_path_list)

    return (target_name_list, target_path_list, atlas_img_path_superlist, atlas_lab_path_superlist, target_mask_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segments target images with label fusion methods.\n'
                                                 'Atlas parameters are optional. In not given, then leave-one-out\n'
                                                 'label fusion is done among target images (target labels must be given).\n'
                                                 'Patch size and search radius can be specified')
    # COMMON ARGUMENTS
    parser.add_argument("--target_dir", type=str, nargs=1, required=True, help='directory of target images')
    parser.add_argument("--target_img_suffix", type=str, nargs=1, required=True, help='image suffixes')
    parser.add_argument("--target_lab_suffix", type=str, nargs=1, help="(optional) in case leave-one-out label fusion")
    parser.add_argument("--target_mask_suffix", type=str, nargs=1, help="(optional) suffix of mask where to perform label fusion")

    parser.add_argument("--labfus_dir", type=str, nargs=1, help='(optional) directory to store label fusion results')

    parser.add_argument("--atlas_selection", type=str, nargs=4, help="(optional) number of atlases, scores file, "
                                                                     "suffix inside scores file for target, idem for atlas")

    parser.add_argument("--out_reg_dir", type=str, nargs=1, required=True, help='registrations directory')
    parser.add_argument("--num_procs", type=int, nargs=1, default=[8], help='number of concurrent processes ')

    # ONLY IF OUT_REG_DIR EXISTS
    parser.add_argument("--reg_img_suffix", type=str, nargs=1, help='if out_reg_dir exists, suffix of warped img files')
    parser.add_argument("--reg_lab_suffix", type=str, nargs=1, help='if out_reg_dir exists, suffix of warped label files')

    # ONLY IF OUT_REG_DIR DOES NOT EXIST
    parser.add_argument("--target_reg_dir", type=str, nargs=1, help='directory with target transformations to template space')
    parser.add_argument("--target_linear_intfix", type=str, nargs=1, help="intfix of the input linear transform")
    parser.add_argument("--target_deform_intfix", type=str, nargs=1, help="(optional) intfix of the input deformation field")

    parser.add_argument("--atlas_dir", type=str, nargs=1, help="(optional) in case not LOO provide atlas directory")
    parser.add_argument("--atlas_img_suffix", type=str, nargs=1, help="(optional) same as target_img_suffix if not provided")
    parser.add_argument("--atlas_lab_suffix", type=str, nargs=1, help="(optional) same as target_lab_suffix if not provided")

    parser.add_argument("--atlas_reg_dir", type=str, nargs=1, help="(optional) same as target_reg_dir if not provided")
    parser.add_argument("--atlas_linear_intfix", type=str, nargs=1, help="(optional) same as target_linear_intfix if not provided")
    parser.add_argument("--atlas_deform_intfix", type=str, nargs=1, help="(optional) same as target_deform_intfix if not provided")

    parser.add_argument("--template_file", type=str, nargs=1, help="(optional) do label fusion in template space")
    parser.add_argument("--keep_reg_dir", action="store_true", help='keep temp directory (for debug)')
    parser.add_argument("--maskout_reg_files", action="store_true", help='maskout registered images with target_mask (to save space)')

    # LABFUS ARGUMENTS
    parser.add_argument("--target_parallelism", action="store_true", help="parallelize label fusion along target images (faster but takes more disk space) (use only when small number of targets/atlases)")
    parser.add_argument("--probabilities", action="store_true", help="compute segmentation probabilities")
    parser.add_argument("--float", action="store_true", help='use single precision computations')
    parser.add_argument("--num_itk_threads", type=int, nargs=1, default=[1], help='number of threads per ANTs proc ')

    parser.add_argument("--patch_rad", type=str, nargs=1, help="patch radius #x#x# (joint, nlwv, lasso, nlbeta, deeplf)")
    parser.add_argument("--search_rad", type=str, nargs=1, help="search radius #x#x# (joint, nlwv, lasso, nlbeta, deeplf)")
    parser.add_argument("--fusion_rad", type=str, nargs=1, help="fusion radius #x#x# (nlwv, lasso, nlbeta, deeplf)")
    parser.add_argument("--struct_sim", type=float, nargs=1, help="structural similarity threshold (nlwv, lasso, nlbeta, deeplf)")
    parser.add_argument("--patch_norm", type=str, nargs=1, help="patch normalization [zscore | l2 | zl2 | none] (nlwv, lasso, nlbeta, myjoint, deeplf)")
    parser.add_argument("--metric", type=str, nargs=1, default=['sqeuclidean'], help="metric for patch comparisons (default sqeuclidean)")

    parser.add_argument("--nlwv_suffix", type=str, nargs=1, help="nlwv suffix to be added to the target image name without target_img_suffix")
    parser.add_argument("--nlwv_h_list", type=float, nargs='+', default=[0.01], help="std (nlwv)")

    parser.add_argument("--lasso_suffix", type=str, nargs=1, help="lasso suffix to be added to the target image name without target_img_suffix")
    parser.add_argument("--lasso_lbd_list", type=float, nargs='+', default=[0.001], help="sparsity (lasso)")

    parser.add_argument("--nlbeta_suffix", type=str, nargs=1, help="nlbeta suffix to be added to the target image name without target_img_suffix")
    parser.add_argument("--nlbeta_b_list", type=float, nargs='+', default=[0.1], help="scale (nlbeta)")
    # parser.add_argument("--nlbeta_with_deeplf_parms", action="store_true", help="get nlbeta params from deep net label fusion params")

    parser.add_argument("--deeplf_suffix", type=str, nargs=1, help="deeplf suffix to be added to the target image name without target_img_suffix")
    parser.add_argument("--deeplf_net_list", type=str, nargs='+', help="network files (deeplf)")

    parser.add_argument("--myjoint_suffix", type=str, nargs=1, help="myjoint suffix to be added to the target image name without target_img_suffix")
    parser.add_argument("--myjoint_h_list", type=float, nargs='+', default=[3.0], help="param (myjoint)")

    parser.add_argument("--joint_suffix", type=str, nargs=1, help="joint suffix to be added to the target image name without target_img_suffix")
    parser.add_argument("--joint_alpha", type=float, nargs=1, help="value for alpha in joint label fusion")
    parser.add_argument("--joint_beta", type=float, nargs=1, help="value for alpha in joint label fusion")
    parser.add_argument("--joint_metric", type=str, nargs=1, help="similarity metric in joint label fusion: PC (pearson, default), MSQ (mean squares)")

    parser.add_argument("--majvot_suffix", type=str, nargs=1, help="majvot suffix to be added to the target image name without target_img_suffix")
    # parser.add_argument("--staple_suffix", type=str, nargs=1, help="staple suffix to be added to the target image name without target_img_suffix")

    parser.add_argument("--parallel_labels_list", action='append', type=int, nargs='+', help="(append) list of labels to be paired with parameter list")

    args = parser.parse_args()
    # args = parser.parse_args('--target_dir /home/sanromag/DATA/OB/data_partitions/kk/ '
    #                          '--target_img_suffix _t2.nii.gz '
    #                          '--target_lab_suffix _OBV.nii.gz '
    #                          '--target_mask_suffix _mask.nii.gz '
    #                          '--atlas_selection 15 /home/sanromag/DATA/OB/templates/NormCorr_S3mXtpl.dat _t2_S3mXtpl_Warped.nii.gz _t2_S3mXtpl_Warped.nii.gz '
    #                          '--out_reg_dir /home/sanromag/DATA/OB/data_partitions/reg_S3m '
    #                          '--reg_img_suffix _t2Warped.nii.gz '
    #                          '--reg_lab_suffix _OBVWarped.nii.gz '
    #                          '--keep_reg_dir '
    #                          '--num_procs 5 '
    #                          '--target_parallelism '
    #                          '--float '
    #                          '--patch_rad 2x2x2 '
    #                          '--search_rad 1x1x1 '
    #                          '--joint_suffix _joint.nii.gz '
    #                          '--joint_alpha 1e-7 '
    #                          '--joint_beta 4.0 '
    #                          '--labfus_dir /home/sanromag/DATA/OB/labfus/kk '.split())


    # Check if reg dir exists

    reg_dir_exists = False
    if os.path.exists(args.out_reg_dir[0]):
        reg_dir_exists = True

    do_labfus_using_existing_regdir = reg_dir_exists and args.reg_img_suffix is not None and args.reg_lab_suffix is not None


    if do_labfus_using_existing_regdir:

        # if want to use existing regdir -> gather filenames

        target_mask_suffix = None
        if args.target_mask_suffix is not None:
            target_mask_suffix = args.target_mask_suffix[0]

        aux = get_label_fusion_params(args.target_dir[0], args.target_img_suffix[0], target_mask_suffix, args.out_reg_dir[0],
                                      args.reg_img_suffix[0], args.reg_lab_suffix[0], args.atlas_selection)
        target_name_list, target_path_list, atlas_img_path_superlist, atlas_lab_path_superlist, target_mask_files = aux
        Ntar = len(target_name_list)

        # print('NUMBER OF TARGETS %d' % (Ntar))
        # print('%s' % (target_name_list))
        # exit()

    else:

        #
        # else -> create and warp everything

        if not reg_dir_exists: os.makedirs(args.out_reg_dir[0])

        files_list = os.listdir(args.target_dir[0])
        target_img_files = [f for f in files_list if f.endswith(args.target_img_suffix[0])]
        assert target_img_files, "List of target files is empty"
        Ntar = len(target_img_files)

        if args.target_lab_suffix is not None:
            target_lab_files = [f.split(args.target_img_suffix[0])[0] + args.target_lab_suffix[0] for f in target_img_files]
            assert False not in [os.path.exists(os.path.join(args.target_dir[0], f)) for f in target_lab_files], "Some target label file not found"

        if args.target_mask_suffix is not None:
            target_mask_files = [f.split(args.target_img_suffix[0])[0] + args.target_mask_suffix[0] for f in target_img_files]
            assert False not in [os.path.exists(os.path.join(args.target_dir[0], f)) for f in target_mask_files], "Some target mask file not found"

        is_loo = False
        if args.atlas_dir is not None:
            files_list = os.listdir(args.atlas_dir[0])
            atlas_dir = args.atlas_dir[0]
            atlas_img_suffix = args.atlas_img_suffix[0] if args.atlas_img_suffix is not None else args.target_img_suffix[0]
            atlas_lab_suffix = args.atlas_lab_suffix[0] if args.atlas_lab_suffix is not None else args.target_lab_suffix[0]
            atlas_reg_dir = args.atlas_reg_dir[0] if args.atlas_reg_dir is not None else args.target_reg_dir[0]
            atlas_linear_intfix = args.atlas_linear_intfix[0] if args.atlas_linear_intfix is not None else args.target_linear_intfix[0]
            atlas_deform_intfix = None
            if args.atlas_deform_intfix is not None or args.target_deform_intfix is not None:
                atlas_deform_intfix = args.atlas_deform_intfix[0] if args.atlas_deform_intfix is not None else args.target_deform_intfix[0]
            atlas_img_files = [f for f in files_list if f.endswith(atlas_img_suffix)]
            assert atlas_img_files, "List of atlas files is empty"
            atlas_lab_files = [f.split(atlas_img_suffix)[0] + atlas_lab_suffix for f in atlas_img_files]
            assert False not in [os.path.exists(os.path.join(args.atlas_dir[0], f)) for f in atlas_lab_files], "Some target label file not found"
            Natl = len(atlas_img_files)
        else:
            print "Leave one out segmentation"
            # print('target images: %s' % (target_img_files))
            is_loo = True
            atlas_img_files = target_img_files
            atlas_lab_files = target_lab_files
            Natl = Ntar
            atlas_dir = args.target_dir[0]
            atlas_img_suffix = args.target_img_suffix[0]
            atlas_lab_suffix = args.target_lab_suffix[0]
            atlas_reg_dir = args.target_reg_dir[0]
            atlas_linear_intfix = args.target_linear_intfix[0]
            atlas_deform_intfix = None
            if args.target_deform_intfix is not None:
                atlas_deform_intfix = args.target_deform_intfix[0]

        #
        # atlas selection

        if args.atlas_selection is not None:

            target_names = [f.split(args.target_img_suffix[0])[0] for f in target_img_files]
            atlas_names = target_names
            if not is_loo:
                atlas_names = [f.split(atlas_img_suffix)[0] for f in atlas_img_files]

            atlas_idx = atlas_selection(args.atlas_selection, target_names, atlas_names, is_loo)

        else:

            print('No atlas selection')

            if is_loo:
                atlas_idx = np.array([list(set(range(Ntar))-{i}) for i in range(Ntar)])
            else:
                atlas_idx = np.array([range(Natl),] * Ntar)


        #
        # if template file -> warp target & atlas to template

        # Label fusion in template space
        if args.template_file is not None:

            cmdline = ['python', '-u', warptotemplate_path]
            cmdline += ['--in_dir', args.target_dir[0]]
            cmdline += ['--linear_suffix', args.target_img_suffix[0]]
            if args.target_lab_suffix is not None:
                cmdline += ['--nearest_suffix', args.target_lab_suffix[0]]
            cmdline += ['--template_file', args.template_file[0]]
            cmdline += ['--reg_dir', args.target_reg_dir[0]]
            cmdline += ['--in_linear_intfix', args.target_linear_intfix[0]]
            if args.target_deform_intfix is not None:
                cmdline += ['--in_deform_intfix', args.target_deform_intfix[0]]
            cmdline += ['--out_dir', args.out_reg_dir[0]]
            cmdline += ['--out_suffix', '_Warped.nii.gz']
            if args.float: cmdline += ['--float']
            cmdline += ['--num_procs', '%d' % args.num_procs[0]]

            print "Warping targets to template"

            call(cmdline)

            if not is_loo:

                cmdline = ['python', '-u', warptotemplate_path]
                cmdline += ['--in_dir', args.atlas_dir[0]]
                cmdline += ['--linear_suffix', atlas_img_suffix]
                cmdline += ['--nearest_suffix', atlas_lab_suffix]
                cmdline += ['--template_file', args.template_file[0]]
                cmdline += ['--reg_dir', atlas_reg_dir]
                cmdline += ['--in_linear_intfix', atlas_linear_intfix]
                if atlas_deform_intfix is not None:
                    cmdline += ['--in_deform_intfix', atlas_deform_intfix]
                cmdline += ['--out_dir', args.out_reg_dir[0]]
                cmdline += ['--out_suffix', '_Warped.nii.gz']
                if args.float: cmdline += ['--float']
                cmdline += ['--num_procs', '%d' % args.num_procs[0]]

                print "Warping atlases to template"

                call(cmdline)

    #
    # Loop over target images
    #

    launcher = Launcher(args.num_procs[0])

    method_exists = False

    for i_t in range(Ntar):

        if do_labfus_using_existing_regdir:

            # If want to do labfus with existing regdir -> pick params for current target file

            target_name = target_name_list[i_t]
            target_path = target_path_list[i_t]
            atlas_img_path_list = atlas_img_path_superlist[i_t]
            atlas_lab_path_list = atlas_lab_path_superlist[i_t]

            target_mask_path = None
            if args.target_mask_suffix is not None:
                target_mask_path = os.path.join(args.target_dir[0], target_mask_files[i_t])

        else:

            target_name = target_img_files[i_t].split(args.target_img_suffix[0])[0]

            # Label fusion in target space
            if args.template_file is None:

                #
                # if reg dir doesnt exist and not in template space -> warp atlases to current target

                target_reg_dir = os.path.join(args.out_reg_dir[0], target_name)

                os.makedirs(target_reg_dir)

                for i_a in atlas_idx[i_t]:
                    os.symlink(os.path.join(atlas_dir, atlas_img_files[i_a]), os.path.join(target_reg_dir, atlas_img_files[i_a]))
                    os.symlink(os.path.join(atlas_dir, atlas_lab_files[i_a]), os.path.join(target_reg_dir, atlas_lab_files[i_a]))

                cmdline = ['python', '-u', warpatlasestotarget_path]
                cmdline += ['--atlas_dir', target_reg_dir]
                cmdline += ['--atlas_linear_suffix', atlas_img_suffix]
                cmdline += ['--atlas_nearest_suffix', atlas_lab_suffix]
                cmdline += ['--atlas_reg_dir', atlas_reg_dir]
                cmdline += ['--atlas_linear_intfix', atlas_linear_intfix]
                if atlas_deform_intfix is not None:
                    cmdline += ['--atlas_deform_intfix', atlas_deform_intfix]
                cmdline += ['--target_file', os.path.join(args.target_dir[0], target_img_files[i_t])]
                cmdline += ['--target_suffix', args.target_img_suffix[0]]
                cmdline += ['--target_reg_dir', args.target_reg_dir[0]]
                cmdline += ['--target_linear_intfix', args.target_linear_intfix[0]]
                if args.target_deform_intfix is not None:
                    cmdline += ['--target_deform_intfix', args.target_deform_intfix[0]]
                cmdline += ['--out_dir', target_reg_dir]
                cmdline += ['--out_suffix', '_Warped.nii.gz']
                if args.float: cmdline += ['--float']
                cmdline += ['--num_procs', '%d' % args.num_procs[0]]

                print "Warping atlases to target {}".format(target_img_files[i_t])

                call(cmdline)

                target_path = os.path.join(args.target_dir[0], target_img_files[i_t])
                atlas_img_path_list = [os.path.join(target_reg_dir, atlas_img_files[i_a].split(os.extsep, 1)[0] + '_Warped.nii.gz') for i_a in atlas_idx[i_t]]
                atlas_lab_path_list = [os.path.join(target_reg_dir, atlas_lab_files[i_a].split(os.extsep, 1)[0] + '_Warped.nii.gz') for i_a in atlas_idx[i_t]]

                target_mask_path = None
                if args.target_mask_suffix is not None:
                    target_mask_path = os.path.join(args.target_dir[0], target_mask_files[i_t])

                    # If there's target mask offer the option of masking out atlases to save space
                    if args.maskout_reg_files:
                        cmdline = ['python', '-u', maskout_path]
                        cmdline += ['--in_dir', target_reg_dir]
                        cmdline += ['--in_suffix_list', atlas_img_suffix.split(os.extsep, 1)[0] + '_Warped.nii.gz']
                        cmdline += ['--mask_file', target_mask_path]
                        cmdline += ['--num_procs', '%d' % args.num_procs[0]]

                        print "Masking out atlases of target {}".format(target_img_files[i_t])

                        call(cmdline)


            else:

                target_path = os.path.join(args.out_reg_dir[0], target_img_files[i_t].split(os.extsep, 1)[0] + '_Warped.nii.gz')
                atlas_img_path_list = [os.path.join(args.out_reg_dir[0], atlas_img_files[i_a].split(os.extsep, 1)[0] + '_Warped.nii.gz') for i_a in atlas_idx[i_t]]
                atlas_lab_path_list = [os.path.join(args.out_reg_dir[0], atlas_lab_files[i_a].split(os.extsep, 1)[0] + '_Warped.nii.gz') for i_a in atlas_idx[i_t]]

                target_mask_path = None
                if args.target_mask_suffix is not None:
                    target_mask_path = os.path.join(args.target_dir[0], target_mask_files[i_t])

        #
        # Label fusion

        print("Launching label fusion of file %s" % target_name)

        suffix_list = [args.nlwv_suffix, args.lasso_suffix, args.nlbeta_suffix, args.myjoint_suffix, args.deeplf_suffix, args.joint_suffix, args.majvot_suffix]#, args.staple_suffix]

        method_exists = np.any(np.array([f is not None for f in suffix_list], dtype=np.bool))

        if method_exists:

            assert args.labfus_dir is not None, 'Need to provide labfus_dir when running some method'

            triplets_list = zip(['nlwv', 'lasso', 'nlbeta', 'myjoint', 'deeplf', 'joint', 'majvot', 'staple'],
                                [args.nlwv_h_list, args.lasso_lbd_list, args.nlbeta_b_list, args.myjoint_h_list, args.deeplf_net_list, None, None, None],
                                suffix_list)

            out_dir = args.labfus_dir[0]
            if not os.path.exists(out_dir): os.makedirs(out_dir)

            jobs_list = []

            for method, params_list, suffix in triplets_list:

                if suffix is not None:

                    out_file = os.path.join(out_dir, target_name + suffix[0])

                    patch_rad = args.patch_rad[0] if args.patch_rad is not None else None
                    search_rad = args.search_rad[0] if args.search_rad is not None else None
                    fusion_rad = args.fusion_rad[0] if args.fusion_rad is not None else None
                    struct_sim = args.struct_sim[0] if args.struct_sim is not None else None
                    patch_norm = args.patch_norm[0] if args.patch_norm is not None else None

                    joint_alpha = args.joint_alpha[0] if args.joint_alpha is not None else None
                    joint_beta = args.joint_beta[0] if args.joint_beta is not None else None
                    joint_metric = args.joint_metric[0] if args.joint_metric is not None else None

                    launcher = label_fusion(launcher, target_path, atlas_img_path_list, atlas_lab_path_list, out_file, args.probabilities, method, args.metric[0], target_mask_path,
                                            patch_rad, search_rad, fusion_rad, struct_sim, patch_norm, params_list, joint_alpha, joint_beta, joint_metric,
                                            args.parallel_labels_list, args.num_itk_threads[0])

            if not args.target_parallelism:
                launcher.wait()

            # Remove warped files
            if not reg_dir_exists and not args.keep_reg_dir and not args.target_parallelism and args.template_file is None:
                rmtree(target_reg_dir)

        else:

            print('There is not method to execute')


    # Wait for the jobs to finish (in cluster)
    if method_exists and args.target_parallelism:
        print("Waiting for label fusion jobs to finish...")
        launcher.wait()

    print("Label fusion finished.")

    if not reg_dir_exists and not args.keep_reg_dir:
        rmtree(args.out_reg_dir[0])





