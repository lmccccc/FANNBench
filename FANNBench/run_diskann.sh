export debugSearchFlag=0
#! /bin/bash

source ./vars.sh $1 $2 $3 $4
source ./file_check.sh

algo=DiskANN
##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")


dir=logs/${now}_${dataset}_${algo}

if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$diskann_index_label_root" ]; then
    mkdir ${diskann_root}
    mkdir ${diskann_index_root}
    mkdir ${diskann_index_label_root}
    mkdir ${diskann_result_root}
fi


if [ -e $dataset_bin_file ]; then
    echo "dataset bin already exist"
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
        echo "Python script json2txt.py failed with exit status $status"
        exit $status
    else
        echo "Python script json2txt.py ran successfully"
    fi
fi

if [ -e $keyword_query_range_file ]; then
    echo "query keyword file already exist"
else
    echo "convert json range query label to keyword txt"
    python utils/range2keyword.py $query_range_file $keyword_query_range_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script range2keyword.py failed with exit status $status"
        exit $status
    else
        echo "Python script range2keyword.py ran successfully"
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



log_file=${dir}/summary_${algo}_${dataset}_L${L}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

# mkdir ${dir}

# TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${algo}_${dataset}.txt

if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $diskann_index_file ]; then
        echo "index file already exist"
        exit 0
    else
        echo  "construct index"
        
        echo "dataset file: $dataset_bin_file" 
        echo "label file: $label_file"
        /bin/time -v -p ../DiskANN/build/apps/build_memory_index  --data_type float \
                                                --dist_fn l2 \
                                                --data_path $dataset_bin_file \
                                                --index_path_prefix $diskann_index_file \
                                                -R $M \
                                                --alpha $alpha \
                                                --label_file $label_file \
                                                -T $threads \
                                                &>> $log_file
        if [ $? -ne 0 ]; then
            echo "Diskann constructor failed to run."
        else
            echo "DIskann index constructed."
        fi
    fi
fi

if [ "$mode" == "query" ] || [ "$mode" == "all" ]; then
    # --universal_label $universal_label \
    echo "index file: $diskann_index_file" 
    echo "query file: $query_bin_file"
    echo "ground truth file: $ground_truth_bin_file"
    echo "query label file: $keyword_query_range_file"
    echo "result save path: $diskann_result_path"
    /bin/time -v -p ../DiskANN/build/apps/search_memory_index  --data_type float \
                                            --dist_fn l2 \
                                            --index_path_prefix $diskann_index_file \
                                            --query_file $query_bin_file \
                                            --gt_file $ground_truth_bin_file \
                                            --query_filters_file $keyword_query_range_file \
                                            -K $K \
                                            -L $L \
                                            --result_path $diskann_result_path \
                                            -T $threads \
                                            &>> $log_file
    echo "../DiskANN/build/apps/search_memory_index  --data_type float  --dist_fn l2  --index_path_prefix $diskann_index_file  --query_file $query_bin_file  --gt_file $ground_truth_bin_file --query_filters_file $keyword_query_range_file -K $K -L $L --result_path $diskann_result_path -T $threads &>> $log_file"
    status=$?
    if [ $status -ne 0 ]; then
        echo "diskann query failed with exit status $status"
    else
        echo "diskann query ran successfully"
    fi
fi

if [ $status -eq 0 ]; then
    source ./run_txt2csv.sh
fi