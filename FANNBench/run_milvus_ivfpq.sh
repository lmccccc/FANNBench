export debugSearchFlag=0
#! /bin/bash

source ./vars.sh $1 $2 $3
source ./file_check.sh



algo=Milvus_IVFPQ

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")



dir=logs/${now}_${dataset}_${algo}


if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

log_file=${dir}/summary_${algo}_${dataset}_nprobe${nprobe}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file


# echo nprobe: $nprobe
if [ "$label_cnt" -gt 1 ]; then
    /bin/time -v -p python -u utils/pymilvus_ivfpq_keyword.py $dataset \
                                    $N \
                                    $dataset_file \
                                    $query_file \
                                    $dataset_attr_file \
                                    $query_range_file \
                                    $ground_truth_file \
                                    $K \
                                    $nprobe \
                                    $collection_name \
                                    $mode \
                                    $partition_size_M \
                                    $dim \
                                    &>> $log_file
else
    /bin/time -v -p python -u utils/pymilvus_ivfpq.py $dataset \
                                    $N \
                                    $dataset_file \
                                    $query_file \
                                    $dataset_attr_file \
                                    $query_range_file \
                                    $ground_truth_file \
                                    $K \
                                    $nprobe \
                                    $collection_name \
                                    $mode \
                                    $partition_size_M \
                                    $dim \
                                    &>> $log_file
fi


status=$?
if [ $status -ne 0 ]; then
    echo "milvus ivfpq failed with exit status $status"
    exit $status
else
    echo "milvus ivfpq ran successfully"
fi

source ./run_txt2csv.sh