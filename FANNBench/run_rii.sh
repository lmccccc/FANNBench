export debugSearchFlag=0
#! /bin/bash

source ./vars.sh
source ./file_check.sh



algo=rii

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")



dir=${now}_${dataset}_${algo}
mkdir ${dir}
mkdir ${rii_root}
mkdir ${rii_index_root}


TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${algo}_${datset}.txt

python -u utils/test_rii.py $N $dataset_file $query_file $train_file $dataset_attr_file $query_range_file $ground_truth_file $partition_size_M $K $rii_index_file  &>> ${dir}/summary_${algo}_${datset}.txt