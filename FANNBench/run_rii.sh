export debugSearchFlag=0
#! /bin/bash

source ./vars.sh $1
source ./file_check.sh



algo=RII

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")



dir=logs/${now}_${dataset}_${algo}
if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$rii_index_root" ]; then
    mkdir ${rii_root}
    mkdir ${rii_index_root}
fi

log_file=${dir}/summary_${algo}_${dataset}.txt

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${algo}_${dataset}.txt

echo "N: $N"
echo "dataset file: $dataset_file"
echo "query file: $query_file"
echo "train file: $train_file"
echo "dataset attr file: $dataset_attr_file"
echo "query range file: $query_range_file"
echo "ground truth file: $ground_truth_file"
echo "partition size: $partition_size_M"
echo "K: $K"
echo "index file: $rii_index_file"
echo "mode: $mode"
/bin/time -v -p python -u utils/test_rii.py $N \
                            $dataset_file \
                            $query_file \
                            $train_file \
                            $dataset_attr_file \
                            $query_range_file \
                            $ground_truth_file \
                            $partition_size_M \
                            $K \
                            $rii_index_file \
                            $mode \
                            $label_cnt \
                            &>> $log_file

if [ $? -ne 0 ]; then
    echo "rii failed to run."
else
    echo "rii succeed to run."
fi

status=$?
if [ $status -eq 0 ]; then
    source ./run_txt2csv.sh
fi