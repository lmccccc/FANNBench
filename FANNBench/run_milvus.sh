export debugSearchFlag=0
#! /bin/bash

source ./vars.sh
source ./file_check.sh



algo=Milvus

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")



dir=logs/${now}_${dataset}_${algo}
mkdir ${dir}


TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${algo}_${dataset}.txt

python -u utils/test_pymilvus.py $dataset \
                                 $N \
                                 $dataset_file \
                                 $query_file \
                                 $dataset_attr_file \
                                 $query_range_file \
                                 $ground_truth_file \
                                 $K \
                                 $nprobe \
                                 $label_attr \
                                 &>> ${dir}/summary_${algo}_${dataset}.txt