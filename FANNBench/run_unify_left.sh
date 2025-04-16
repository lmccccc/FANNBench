export debugSearchFlag=0
#! /bin/bash

source ./vars.sh $1 $2 $3 $4 $5
source ./file_check.sh



algo=UNIFY_left

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

dir=logs/${now}_${dataset}_${algo}

if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$unify_result_root" ]; then
    mkdir ${unify_root}
    mkdir ${unify_index_root}
    mkdir ${unify_result_root}
fi

if [ -f "$unify_result_file" ]; then
    echo "Remove exist result file"
    rm $unify_result_file
fi

log_file=${dir}/summary_${algo}_${dataset}_${ef_search}_${AL}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

echo "dataset: $dataset"
echo "datasize: $N"
echo "query_size: $query_size"
echo "dataset_file: $dataset_file"
echo "query_file: $query_file"
echo "dataset_attr_file: $dataset_attr_file"
echo "query_range_file: $query_range_file"
echo "ground_truth_file: $ground_truth_file"
echo "unify_index_file: $unify_index_file"
echo "unify_result_file: $unify_result_file"
echo "top_k: $K"
echo "threads: $threads"
echo "ef_search: $ef_search"

if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $unify_index_file ]; then
        echo "index file already exist at $unify_index_file"
        exit 0
    fi
fi

/bin/time -v -p python -u utils/search_hsig.py --use_mbv_hnsw true \
                                               --data_path $dataset_file \
                                               --index_cache_path $unify_index_file \
                                               --result_save_path $unify_result_file \
                                               --B $B_unify \
                                               --ef_list $ef_search \
                                               --al_list $AL \
                                               --k $K \
                                               --N $N \
                                               --M $M \
                                               --efConstruction $ef_construction \
                                               --mode $mode \
                                               --query_path $query_file \
                                               --attr_path $dataset_attr_file \
                                               --qrange_path $query_range_file \
                                               --gt_path $ground_truth_file \
                                               --n_query_to_use $query_size \
                                               --threads $threads \
                                               &>> $log_file

if [ $? -ne 0 ]; then
    echo "UNIFY failed to run."
else
    echo "UNIFY succeed."
fi

status=$?
if [ $status -eq 0 ]; then
    source ./run_txt2csv.sh
fi