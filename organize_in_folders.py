import os
import argparse
import sys

parser = argparse.ArgumentParser('Puts modalities for each subject in a folder with same ID')
parser.add_argument("--in_dir", type=str, nargs=1, required=True, help="directory containing all files from all subjects")
parser.add_argument("--in_suffix_list", type=str, nargs='+', required=True, help="list of suffixes for each modality")
parser.add_argument("--out_dir", type=str, nargs=1, required=True, help="base directory where to create the out directory structure")
parser.add_argument("--out_name_list", type=str, nargs='+', help="(optional) list of names to give to output files (if not given, same as original names)")
parser.add_argument("--abspath", type='store_true', help="whether to use absolute path for symlinks (default: relative from \'in_dir\')")

args = parser.parse_args()

sys.path.insert(0, os.path.join(os.environ['HOME'], 'CODE', 'src', 'modules'))
from utils import get_files_superlist

names_list, files_superlist, files_superlist_t = get_files_superlist(args.in_dir, args.in_suffix_list)
assert names_list, "filenames list is empty"

if args.out_name_list is not None:
    assert len(args.out_name_list) == len(args.in_suffix_list), 'length suffix list must be equal to out names list'

# create output directory
assert not os.path.exists(args.out_dir[0]), 'error: output directory exists'
os.makedirs(args.out_dir[0])

# create subdirs with IDs
for name in names_list:
    os.makedirs(os.path.join(args.out_dir[0], name))

# change to output directory (in case not absolute paths)
if not args.abspath:
    os.chdir(args.out_dir[0])

# link each subject files to the output subject subdirectory
for name, files_list in zip(names_list, files_superlist_t):
    subject_dir = os.path.join(args.out_dir[0], name)
    for i, file in enumerate(files_list):
        src = os.path.join(args.in_dir[0], file)
        if not args.abspath:
            src = os.path.relpath(os.path.join(args.in_dir[0], file))
        if args.out_name_list is not None:
<<<<<<< HEAD
            os.symlink(src, os.path.join(subject_dir, args.out_name_list[i]))
=======
            out_folder = os.path.dirname(args.out_name_list[i])
            out_name = os.path.basename(args.out_name_list[i])
            if not (out_folder):  # if there is no subfolder, then directly create file
                os.symlink(os.path.join(args.in_dir[0], file), os.path.join(subject_dir, out_name))
            else:  # otherwise create subfolder and link the file within
                os.makedirs(os.path.join(subject_dir, out_folder))
                os.symlink(os.path.join(args.in_dir[0], file), os.path.join(subject_dir, out_folder, out_name))
>>>>>>> df51b05b07349fa39c691cc9c566b9d8d6c9eecb
        else:
            os.symlink(src, os.path.join(subject_dir, file))

