export debugSearchFlag=0
#! /bin/bash

source ./vars.sh construction $1 $2

now=$(date +"%m-%d-%Y")
algo=groundtruth

dir=logs/${now}_${dataset}_${algo}
mkdir ${dir}

#check if succicient file exist
if [ ! -f "$dataset_file" ]; then  
    echo "error, $dataset_file does not exist."
    exit 1
fi
if [ ! -f "$query_file" ]; then  
    echo "error, $query_file does not exist."
    exit 1
fi
if [ ! -f "$dataset_attr_file" ]; then  
    echo "error, $dataset_attr_file does not exist."
    exit 1
fi
if [ ! -f "$query_range_file" ]; then  
    echo "error, $query_range_file does not exist."
    exit 1
fi

if [ -d "$ground_truth_file" ]; then
    exho Ground truth file ${ground_truth_file} already exist. 
    exit 1
fi

if [ "$label_cnt" -gt 1 ]; then
    echo "generate gt for keyword query"
    echo "N: $N"
    echo "dataset_file: $dataset_file"
    echo "dataset_attr_file: $dataset_attr_file"
    echo "query_file: $query_file"
    echo "query_range_file: $query_range_file"
    echo "ground_truth_file: $ground_truth_file"
    echo "gt_topk: $gt_topk"
    echo "dim: $dim"
    /bin/time -v -p ../faiss/build/demos/generate_groundtruth_keyword $N \
                                            $dataset_file \
                                            $dataset_attr_file \
                                            $query_file \
                                            $query_range_file \
                                            $ground_truth_file \
                                            $gt_topk \
                                            $dim \
                                            &>> ${dir}/summary_${algo}_${dataset}.txt 

else
    # <number vecs> <dataset> <attr> <query> <queryrange> <output> <k>
    echo "N: $N"
    echo "dataset_file: $dataset_file"
    echo "dataset_attr_file: $dataset_attr_file"
    echo "query_file: $query_file"
    echo "query_range_file: $query_range_file"
    echo "ground_truth_file: $ground_truth_file"
    echo "gt_topk: $gt_topk"
    echo "dim: $dim"
    /bin/time -v -p ../faiss/build/demos/generate_groundtruth $N \
                                            $dataset_file \
                                            $dataset_attr_file \
                                            $query_file \
                                            $query_range_file \
                                            $ground_truth_file \
                                            $gt_topk \
                                            $dim \
                                            &>> ${dir}/summary_${algo}_${dataset}.txt 
fi

status=$?
if [ $status -ne 0 ]; then
    echo "gt generator failed with exit status $status"
    exit $status
else
    echo "gt generator ran successfully"
fi
