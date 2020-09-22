#!/bin/bash

jobs_fname=jobs_mk_3dROIstats_REST34.txt
readarray -t subj_list < subj_list.txt

touch $jobs_fname

for subj in ${subj_list[@]}; do
    echo "sh 01_mk_3dROIstats.sh ${subj}" >> $jobs_fname
done
