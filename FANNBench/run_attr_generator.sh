export debugSearchFlag=0
#! /bin/bash

now=$(date +"%m-%d-%Y")

source ./vars.sh construction


mkdir ${label_path}

#acorn
# if [ "$label_method" = "keyword" ]; then
#     echo "for keyword"
# elif
#     [ "$label_method" = "range" ]; then
#     echo "for range"
# else
#     echo "method not support"
#     exit 1
# fi


# TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_sift_n=${N}.txt


python utils/attrGenerator.py ${N} \
                              ${dataset_file} \
                              ${dataset_attr_file} \
                              ${label_cnt} \
                              ${label_range} \
                              ${distribution} \
                              ${query_label_cnt} \
                              ${train_file} \
                              ${train_size} \
                              ${centroid_file} \






