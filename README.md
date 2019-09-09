# Neuroimaging Scripts
This repository contains scripts I've developed over the last years for efficient processing of Neuroimaging data.
Some of the scripts require installation of ANTs or FSL tools.

Common Neuroimaging packages provide useful tools to do all kinds of stuff with Neuroimaging data.
However, they are designed to process on a case-by-case basis.
When processing large batches of data, one has therefore to build scripts that run the tools on each file and save the results.

This repository aims at providing these scripts.
They rely on the convention of using a common suffix for files of the same type or function: eg, \_T1.nii.gz, \_mask.nii.gz, ...
Check the script for details on the arguments.

The repository contains the following scripts:
* `average_images.py`: averages images
* `check_headers.py`: Checks the dimensions and orientations are consistent accross modalities of same subject
* `compute_mask.py`: Creates multiple masks (one for each file) or a single consensus mask for all files (if template is provided) 
* `compute_similarities.py`: Computes the similarity between images according to some metric 
* `compute_vol_dice.py`: Plots lesion vol vs. dice
* `create_symlinks.py`: Creates symbolic links of the files from one directory onto another 
* `decide_atlases.py`: Selects one (or multiple) sets of images evenly spread accross the population according to some similarity metric. Useful to select train (atlases) and test (target) sets for cross-validation. 
* `evaluate_segmentations.py`: Computes Dice score of estimated segmentations w.r.t. ground truth segmentations. Average per-label Dice score and average per-subject Dice score are stored in label\_dice.csv and subj\_dice.csv in est\_dir directory, respectively 
* `filter_labels.py`: Selects a subset of labels from list of labelmaps 
* `label_fusion.py`: Workhorse for label fusion using multiple methods. 
* `maskout.py`: Masks-out images given a mask 
* `merge_probmaps.py`: merge probability maps
* `organize_in_folders.py`: Puts modalities for each subject in a folder with same ID
* `process_images.py`: Processes the images including N4 correction and histogram matching to template. Optionally, images and/or template can be masked out (e.g., remove skull) given the mask file. 
* `register_pairs_flirt.py`: Register pairs of images (eg, baseline and follow-up) using FSL flirt 
* `register_to_template_ants.py`: Registers images to template. Can use initial transformation. Uses ANTs.
* `register_to_template_flirt.py`: Registers images to template. Can use initial transformation. Uses FSL Flirt.
* `threshold_probmaps.py`: Thresholds label probability maps and outputs binary segmentations 
* `warp_atlases_to_target.py`: Warp atlas images in a directory to a target image using given linear and/or deformable transforms. Optionally, the target can be warped to atlases with the --inverse flag. 
* `warp_to_template.py`: Warp images to template space. Optionally, the inverse transformation can be done 

## License
License for this software package: BSD 3-Clause License. A copy of this license is present in the root directory
