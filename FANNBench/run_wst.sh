export debugSearchFlag=0
#! /bin/bash

source ./vars.sh $1 $2 $3 $4 $5
source ./file_check.sh



algo=WST_opt

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

dir=logs/${now}_${dataset}_${algo}

if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $rfann_index_prefix ]; then
        echo "index file already exist at $rfann_index_prefix"
        exit 0
    fi
fi


if [ ! -d "$rfann_index_prefix" ]; then
    mkdir ${rfann_root}
    mkdir ${rfann_index_root}
    mkdir ${rfann_index_prefix}
    mkdir ${rfann_result_root}
fi

log_file=${dir}/summary_${algo}_${dataset}_beamsize${beamsize}_finalbeammul${final_beam_multiply}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

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
echo "log file: $log_file"
/bin/time -v -p python -u utils/RangeFilteredANN.py --dataset $dataset \
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
                                 --super_opt_postfiltering_split_factor $split_factor \
                                 --super_opt_postfiltering_shift_factor $shift_factor \
                                 --mode $mode \
                                 &>> $log_file
# $N $dataset_file $query_file $train_file $dataset_attr_file $query_range_file $ground_truth_file $M $K

if [ $? -ne 0 ]; then
    echo "wst super opt post filtering failed to run."
else
    echo "wst super opt post filtering succeed."
fi

status=$?
if [ $status -eq 0 ]; then
    source ./run_txt2csv.sh
fi