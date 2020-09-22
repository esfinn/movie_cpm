#!/bin/bash

module load afni

subj=$@

mask_dir=/data/HCP_preproc/resting_7T_1.6mm_59k_functional_preprocessed_Jan2020
run_dir=/data/HCP_preproc/resting_7T_FIX-Denoised_Dec2019
out_dir=/data/HCP_preproc/7T_movie/SubjectData

mkdir -p $out_dir/$subj

for run in REST1_7T_PA # REST3_7T_PA REST4_7T_AP
do

out=${subj}_${run}_wholebrain_ts.txt

# Create global signal timeseries
3dmaskave -quiet -mask $mask_dir/$subj/MNINonLinear/Results/rfMRI_${run}/brainmask_fs.1.60.nii.gz \
    $run_dir/$subj/MNINonLinear/Results/rfMRI_${run}/rfMRI_${run}_hp2000_clean.nii.gz \
    > $out_dir/$subj/$out

# Regress global signal timeseries from run
3dTproject -input $run_dir/$subj/MNINonLinear/Results/rfMRI_${run}/rfMRI_${run}_hp2000_clean.nii.gz \
    -prefix $out_dir/$subj/rfMRI_${run}_hp2000_clean_gsr.nii.gz \
    -ort $out_dir/$subj/$out \
    -polort -1

done
