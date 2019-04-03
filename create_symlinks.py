import os
import argparse
import sys

parser = argparse.ArgumentParser('Creates symbolic links of the files from one directory onto another')
parser.add_argument("--filenames_dir", type=str, nargs=1, required=True, help='Directory where to get prefixes of filenames to be linked')
parser.add_argument("--filenames_suffix", type=str, nargs=1, required=True, help='Suffix to be removed from filenames to get prefixes')
parser.add_argument("--source_dir", type=str, nargs=1, required=True, help="Source directory to link filenames from (starting with obtained prefixes)")
parser.add_argument("--source_suffix_list", type=str, nargs='+', required=True, help="List of suffixes of files to be linked (to be removed from source files to get matching prefixes)")
parser.add_argument("--destination_dir", type=str, nargs=1, required=True, help="Directory where to link the matched source files onto")
parser.add_argument("--leave_one_out", action='store_true', help='(optional) create sub-directories with filenames for leave-one-out (instead of directly linking all files to dest_dir)')
parser.add_argument("--link_nonmatching", action='store_true', help='(optional) Link only files from source dir that do not match files in filenames dir')
parser.add_argument("--insert_intfix", type=str, nargs=1, help="(optional) Intfix to be inserted before source suffix when creating the link")

args = parser.parse_args()

sys.path.insert(0, os.path.join(os.environ['HOME'], 'CODE', 'src', 'modules'))
from utils import get_files_superlist

filenames_list, _ = get_files_superlist(args.filenames_dir, args.filenames_suffix)
assert filenames_list, "filenames list is empty"

srcnames_list = filenames_list
sources_superlist = []
if not args.link_nonmatching:  # normally link the files in source that match the ones in filenames dir
    sources_superlist = [[f + args.source_suffix_list[i] for i in range(len(args.source_suffix_list))] for f in filenames_list]
    # print [f for sources_list in sources_superlist for f in sources_list]
    # print [os.path.exists(os.path.join(args.source_dir[0], f)) for sources_list in sources_superlist for f in sources_list]
else:  # link only those files in source that do not match to any file in filenames dir
    names_list, files_superlist = get_files_superlist(args.source_dir, args.source_suffix_list)
    srcnames_list = []
    for i, name in enumerate(names_list):
        if name in filenames_list: continue
        srcnames_list.append(name)
        auxfiles_list = []
        for j in range(len(args.source_suffix_list)):
            auxfiles_list.append(files_superlist[j][i])
        sources_superlist.append(auxfiles_list)

aux_list = [f for sources_list in sources_superlist for f in sources_list]
cond_list = [os.path.exists(os.path.join(args.source_dir[0], f)) for f in aux_list]
if False in cond_list:
    print('Following files in filenames_dir not found in source_dir:')
    for i, b in enumerate(cond_list):
        if not b:
            print(aux_list[i])
    sys.exit()
# assert False not in cond_list, "Source dir does not contain all files in filenames dir"

# if destination dir does not exist, create it
if not os.path.exists(args.destination_dir[0]):
    os.makedirs(args.destination_dir[0])

# create sub-directories for LOO
if args.leave_one_out:
    for srcname in srcnames_list:
        if not os.path.exists(os.path.join(args.destination_dir[0], srcname)):
            os.makedirs(os.path.join(args.destination_dir[0], srcname))

for sources_list, srcname in zip(sources_superlist, srcnames_list):
    for i, source_file in enumerate(sources_list):
        # append intfix if needed
        out_file = source_file
        if args.insert_intfix is not None:
            out_file = source_file.split(args.source_suffix_list[i])[0] + args.insert_intfix[0] + args.source_suffix_list[i]
        # LOO?
        if not args.leave_one_out:
            os.symlink(os.path.join(args.source_dir[0], source_file), os.path.join(args.destination_dir[0], out_file))
        else:  # in case LOO link the files to all sub-directories corresponding to different filename
            for srcname2 in srcnames_list:
                if srcname == srcname2:
                    continue
                os.symlink(os.path.join(args.source_dir[0], source_file), os.path.join(args.destination_dir[0], srcname2, source_file))

