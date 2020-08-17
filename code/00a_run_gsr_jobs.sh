#!/bin/bash

jobs_fname=jobs_run_gsr_REST34.txt
readarray -t subj_list < subj_list.txt

touch $jobs_fname

for subj in ${subj_list[@]}; do
    echo "sh 00_run_gsr.sh ${subj}" >> $jobs_fname
done
