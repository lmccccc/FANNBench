export debugSearchFlag=0
#! /bin/bash

now=$(date +"%m-%d-%Y")

source ./vars.sh construction $1 $2

if [ ! -d "$label_path" ]; then
    mkdir ${label_path}
fi

if [ -d "$dataset_attr_file" ]; then
    echo Attribute file ${dataset_attr_file} already exist. 
    exit 1
fi

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






