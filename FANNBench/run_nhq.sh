export debugSearchFlag=0
#! /bin/bash

source ./vars.sh
source ./file_check.sh

algo=nhq
##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")


dir=${now}_${dataset}_${algo}
mkdir ${dir}
mkdir ${nhq_root}
mkdir ${nhq_index_root}
# mkdir ${dir}

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${algo}_${dataset}.txt

if [ -e $keyword_query_range_file ]; then
    echo "query keyword file already exist"
else
    echo "convert json range query label to keyword txt"
    python utils/range2keyword.py $query_range_file $keyword_query_range_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
        exit $status
    else
        echo "Python script ran successfully"
    fi
fi


if [ -e $nhq_index_model_file ]; then
    echo "index file already exist"
else
    echo  "construct index"
    
    echo "dataset file: $dataset_file" 
    echo "label file: $dataset_attr_file"
    ../NHQ-main/NHQ-NPG_nsw/examples/cpp/index $dataset_file \
                                               $dataset_attr_file \
                                               $nhq_index_model_file \
                                               $nhq_index_attr_file \
                                               $M \
                                               $ef_construction \
                                               $N \
                                               $threads \
                                               &>> ${dir}/summary_${algo}_${dataset}.txt
    if [ $? -ne 0 ]; then
        echo "Index constructor failed to run."
        exit 1  # Exit the script with a failure code
    else
        echo "Index constructed."
    fi
fi
# --universal_label $universal_label \
echo "index model file: $nhq_index_model_file" 
echo "index attr file: $nhq_index_attr_file"
echo "query file: $query_file"
echo "ground truth file: $ground_truth_file"
echo "query label file: $keyword_query_range_file"
../NHQ-main/NHQ-NPG_nsw/examples/cpp/search $nhq_index_model_file \
                                            $nhq_index_attr_file \
                                            $query_file \
                                            $ground_truth_file \
                                            $keyword_query_range_file \
                                            $query_size \
                                            $K \
                                            $ef_search \
                                            &>> ${dir}/summary_${algo}_${dataset}.txt