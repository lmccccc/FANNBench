export debugSearchFlag=0
#! /bin/bash

source ./vars.sh

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

# <number vecs> <dataset> <attr> <query> <queryrange> <output> <k>
../faiss/build/demos/generate_groundtruth $N \
                                          $dataset_file \
                                          $dataset_attr_file \
                                          $query_file \
                                          $query_range_file \
                                          $ground_truth_file \
                                          $gt_topk \
                                          &>> ${dir}/summary_${algo}_${dataset}.txt 