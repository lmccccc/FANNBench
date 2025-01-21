export debugSearchFlag=0
#! /bin/bash

source ./vars.sh $1 $2 $3
source ./file_check.sh

algo=NHQ_nsw
##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")


dir=logs/${now}_${dataset}_${algo}

if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$nhq_index_root" ]; then
    mkdir ${nhq_root}
    mkdir ${nhq_index_root}
fi
# mkdir ${dir}

log_file=${dir}/summary_${algo}_${dataset}_weightsearch${weight_search}_efsearch${ef_search}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

if [ -e $keyword_query_range_file ]; then
    echo "query keyword file already exist"
else
    echo "convert json range query label to keyword txt"
    python utils/range2keyword.py $query_range_file $keyword_query_range_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
    else
        echo "Python script ran successfully"
    fi
fi

if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $nhq_index_model_file ]; then
        echo "index file already exist"
        exit 1
    else
        echo  "construct index"
        
        echo "dataset file: $dataset_file" 
        echo "label file: $dataset_attr_file"
        /bin/time -v -p ../NHQ-main/NHQ-NPG_nsw/examples/cpp/index $dataset_file \
                                                $dataset_attr_file \
                                                $nhq_index_model_file \
                                                $nhq_index_attr_file \
                                                $M \
                                                $ef_construction \
                                                $N \
                                                $threads \
                                                &>> $log_file
        if [ $? -ne 0 ]; then
            echo "nhq nsw constructor failed to run."
        else
            echo "nhq nsw constructed."
        fi
    fi
fi

if [ "$mode" == "query" ] || [ "$mode" == "all" ]; then
    # --universal_label $universal_label \
    echo "index model file: $nhq_index_model_file" 
    echo "index attr file: $nhq_index_attr_file"
    echo "query file: $query_file"
    echo "ground truth file: $ground_truth_file"
    echo "query label file: $keyword_query_range_file"
    /bin/time -v -p ../NHQ-main/NHQ-NPG_nsw/examples/cpp/search $nhq_index_model_file \
                                                $nhq_index_attr_file \
                                                $query_file \
                                                $ground_truth_file \
                                                $keyword_query_range_file \
                                                $query_size \
                                                $K \
                                                $ef_search \
                                                &>> $log_file

    if [ $? -ne 0 ]; then
        echo "nhq nsw query constructor failed to run."
        exit 1  # Exit the script with a failure code
    else
        echo "nhq nsw query succeed."
    fi
fi

status=$?
if [ $status -eq 0 ]; then
    source ./run_txt2csv.sh
fi
