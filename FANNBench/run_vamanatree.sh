export debugSearchFlag=0
#! /bin/bash

source ./vars.sh $1 $2 $3
source ./file_check.sh



algo=Vamana_tree

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

dir=logs/${now}_${dataset}_${algo}

if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$vtree_index_prefix" ]; then
    mkdir ${vtree_root}
    mkdir ${vtree_index_root}
    mkdir ${vtree_index_prefix}
    mkdir ${vtree_result_root}
fi

log_file=${dir}/summary_${algo}_${dataset}_beamsize${beamsize}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

echo "dataset: $dataset"
echo "datasize: $N"
echo "query_size: $query_size"
echo "dataset_file: $dataset_file"
echo "query_file: $query_file"
echo "dataset_attr_file: $dataset_attr_file"
echo "query_range_file: $query_range_file"
echo "ground_truth_file: $ground_truth_file"
echo "vtree_result_file: $vtree_result_file"
echo "vtree_index_prefix: $vtree_index_prefix"
echo "top_k: $K"
echo "threads: $threads"
/bin/time -v -p python -u utils/RangeFilteredANN.py --dataset $dataset \
                                 --data_size $N \
                                 --query_size $query_size \
                                 --dataset_file $dataset_file \
                                 --query_file $query_file \
                                 --attr_file $dataset_attr_file \
                                 --qrange_file $query_range_file \
                                 --gt_file $ground_truth_file \
                                 --output_file $vtree_result_file \
                                 --index_prefix $vtree_index_prefix \
                                 --top_k $K \
                                 --vamana_tree \
                                 --threads $threads \
                                 --beam_search_size $beamsize \
                                 --num_final_multiplies $final_beam_multiply \
                                 --super_opt_postfiltering_split_factor $split_factor \
                                 --super_opt_postfiltering_shift_factor $shift_factor \
                                 --mode $mode \
                                 &>> $log_file

if [ $? -ne 0 ]; then
    echo "wst vamana tree failed to run."
    exit 1  # Exit the script with a failure code
else
    echo "wst vamana tree succeed."
fi

source ./run_txt2csv.sh