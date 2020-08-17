#!/bin/bash

module load afni

mask="/data/finnes/ref_templates/shen_1.6mm_268_parcellation_adwarpMNI2009.nii.gz"
run_list=("REST1_7T_PA" "REST3_7T_PA" "REST4_7T_AP")

data_dir=/data/HCP_preproc/7T_movie/SubjectData/
out_dir=/data/HCP_preproc/7T_movie/cpm/data/all_shen268_roi_ts

for subj in "$@"
do

for run in ${run_list[@]}
do

# Without GSR
# 3dROIstats -mask $mask -quiet \
#    $data_dir/$subj/MNINonLinear/Results/rfMRI_${run}/rfMRI_${run}_hp2000_clean.nii.gz \
#    > $out_dir/${subj}_${run}_shen268_roi_ts.txt

# With GSR
3dROIstats -mask $mask -quiet \
    $data_dir/${subj}/rfMRI_${run}_hp2000_clean_gsr.nii.gz \
    > $out_dir/${subj}_${run}_shen268_roi_ts_gsr.txt


done
done
