export debugSearchFlag=0
#! /bin/bash

source ./vars.sh
source ./file_check.sh



algo=RangeFilteredANN

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

dir=${now}_${dataset}_${algo}
mkdir ${dir}
mkdir ${rfann_root}
mkdir ${rfann_index_root}
mkdir ${rfann_index_prefix}
mkdir ${rfann_result_root}

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${algo}_${datset}.txt

echo "dataset: $dataset"
echo "datasize: $N"
echo "query_size: $query_size"
echo "dataset_file: $dataset_file"
echo "query_file: $query_file"
echo "dataset_attr_file: $dataset_attr_file"
echo "query_range_file: $query_range_file"
echo "ground_truth_file: $ground_truth_file"
echo "rfann_result_file: $rfann_result_file"
echo "rfann_index_prefix: $rfann_index_prefix"
echo "top_k: $K"
echo "threads: $threads"
python -u utils/RangeFilteredANN.py --dataset $dataset \
                                 --data_size $N \
                                 --query_size $query_size \
                                 --dataset_file $dataset_file \
                                 --query_file $query_file \
                                 --attr_file $dataset_attr_file \
                                 --qrange_file $query_range_file \
                                 --gt_file $ground_truth_file \
                                 --output_file $rfann_result_file \
                                 --index_prefix $rfann_index_prefix \
                                 --top_k $K \
                                 --super_opt_postfiltering \
                                 --threads $threads \
                                 --beam_search_size $beamsize \
                                 --num_final_multiplies $final_beam_multiply \
                                 &>> ${dir}/summary_${algo}_${datset}.txt
# $N $dataset_file $query_file $train_file $dataset_attr_file $query_range_file $ground_truth_file $M $K