export debugSearchFlag=0
#! /bin/bash

source ./vars.sh
source ./file_check.sh

algo=DiskANN
##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

if [ -e $dataset_bin_file ]; then
    echo "datset bin already exist"
else
    echo "convert base vecs to bin"
    ./../DiskANN/build/apps/utils/fvecs_to_bin float $dataset_file $dataset_bin_file
fi

if [ -e $query_bin_file ]; then
    echo "query bin already exist"
else
    echo "convert query vecs to bin"
    ./../DiskANN/build/apps/utils/fvecs_to_bin float $query_file $query_bin_file
fi

if [ -e $label_file ]; then
    echo "label file already exist"
else
    echo "convert json label to txt"
    python utils/json2txt.py $dataset_attr_file $label_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
        exit $status
    else
        echo "Python script ran successfully"
    fi
fi

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


if [ -e $ground_truth_bin_file ]; then
    echo "groundtruth bin file already exist"
else
    echo "convert json range query label to keyword txt"
    python utils/gt_json2bin.py $ground_truth_file $ground_truth_bin_file $gt_topk

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
        exit $status
    else
        echo "Python script ran successfully"
    fi
fi

dir=logs/${now}_${dataset}_${algo}
mkdir ${dir}
mkdir ${diskann_root}
mkdir ${diskann_index_root}
mkdir ${diskann_index_label_root}
mkdir ${diskann_result_root}
# mkdir ${dir}

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${algo}_${dataset}.txt

if [ -e $diskann_index_file ]; then
    echo "index file already exist"
else
    echo  "construct index"
    
    echo "dataset file: $dataset_bin_file" 
    echo "label file: $label_file"
    ../DiskANN/build/apps/build_memory_index  --data_type float \
                                              --dist_fn l2 \
                                              --data_path $dataset_bin_file \
                                              --index_path_prefix $diskann_index_file \
                                              -R $M \
                                              --alpha $alpha \
                                              --label_file $label_file \
                                              -T $threads \
                                              &>> ${dir}/summary_${algo}_${dataset}.txt
    if [ $? -ne 0 ]; then
        echo "Index constructor failed to run."
        exit 1  # Exit the script with a failure code
    else
        echo "Index constructed."
    fi
fi
# --universal_label $universal_label \
echo "index file: $diskann_index_file" 
echo "query file: $query_bin_file"
echo "ground truth file: $ground_truth_bin_file"
echo "query label file: $keyword_query_range_file"
echo "result save path: $diskann_result_path"
../DiskANN/build/apps/search_memory_index  --data_type float \
                                           --dist_fn l2 \
                                           --index_path_prefix $diskann_index_file \
                                           --query_file $query_bin_file \
                                           --gt_file $ground_truth_bin_file \
                                           --query_filters_file $keyword_query_range_file \
                                           -K $K \
                                           -L $L \
                                           --result_path $diskann_result_path \
                                           -T $threads \
                                           &>> ${dir}/summary_${algo}_${dataset}.txt

source ./run_txt2csv.sh