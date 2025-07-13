export debugSearchFlag=0
#! /bin/bash

source ./vars.sh $1 $2 $3 $4
source ./file_check.sh


# not used in benchmark
algo=Milvus_HNSW

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")



dir=logs/${now}_${dataset}_${algo}
if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi


log_file=${dir}/summary_${algo}_${dataset}_ef_search${ef_search}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

if [ "$label_cnt" -eq 2 ]; then
    /bin/time -v -p python -u utils/pymilvus_hnsw_arbi.py $dataset \
                                    $N \
                                    $dataset_file \
                                    $query_file \
                                    $dataset_attr_file \
                                    $query_range_file \
                                    $ground_truth_file \
                                    $K \
                                    $M \
                                    $ef_construction \
                                    $ef_search \
                                    $label_attr \
                                    $hnsw_collection_name \
                                    $mode \
                                    $dim \
                                    &>> $log_file
elif [ "$label_cnt" -gt 1 ]; then
    # /bin/time -v -p python -u utils/pymilvus_hnsw_keyword.py $dataset \
    #                                 $N \
    #                                 $dataset_file \
    #                                 $query_file \
    #                                 $dataset_attr_file \
    #                                 $query_range_file \
    #                                 $ground_truth_file \
    #                                 $K \
    #                                 $M \
    #                                 $ef_construction \
    #                                 $ef_search \
    #                                 $label_attr \
    #                                 $hnsw_collection_name \
    #                                 $mode \
    #                                 $dim \
    #                                 &>> $log_file
    echo "not supported"
else
    /bin/time -v -p python -u utils/pymilvus_hnsw.py $dataset \
                                    $N \
                                    $dataset_file \
                                    $query_file \
                                    $dataset_attr_file \
                                    $query_range_file \
                                    $ground_truth_file \
                                    $K \
                                    $M \
                                    $ef_construction \
                                    $ef_search \
                                    $label_attr \
                                    $hnsw_collection_name \
                                    $mode \
                                    $dim \
                                    &>> $log_file
fi
status=$?
if [ $status -ne 0 ]; then
    echo "milvus hnsw failed with exit status $status"
else
    echo "milvus hnsw ran successfully"
fi

if [ $status -eq 0 ]; then
    source ./run_txt2csv.sh
fi