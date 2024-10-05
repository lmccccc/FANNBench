export debugSearchFlag=0
#! /bin/bash

now=$(date +"%m-%d-%Y")

source ./vars.sh

# now=$(date +"%m-%d-%Y")
# dir=${now}_${dataset}
# mkdir ${dir}

#sift1M
# N=1000000         # 1M
# query_size=10000           # 10k
# dataset_file="../ACORN/Datasets/sift1M/sift_base.fvecs"
# query_file="../ACORN/Datasets/sift1M/sift_query.fvecs"
# output_dataset_attr_file="../ACORN/testing_data/sift_attr.ivecs"
# output_query_range_file="../ACORN/testing_data/sift_qrange.ivecs"

#sift1B
# N=1000000000         # 1B
# query_size=10000           # 10k
# dataset_file="../../dataset/sift/bigann_base.bvecs"
# query_file="../../dataset/sift/bigann_query.bvecs"
# output_dataset_attr_file="../../dataset/sift/sift_attr.json"
# output_query_range_file="../../dataset/sift/sift_qrange.json"

#acorn
if [ "$query_method" = "keyword" ]; then
    echo "for keyword"
elif
    [ "$query_method" = "range" ]; then
    echo "for range"
else
    echo "method not support"
    exit 1
fi


# TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_sift_n=${N}.txt

python utils/qrangeGenerator.py ${query_size} \
                                ${query_file} \
                                ${query_range_file} \
                                ${query_method} \
                                ${label_range} \
                                ${query_label_cnt} \
                                ${distribution} \
                                ${centroid_file}





