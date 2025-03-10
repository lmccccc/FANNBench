export debugSearchFlag=0
#! /bin/bash

now=$(date +"%m-%d-%Y")

source ./vars.sh construction $1 $2

# now=$(date +"%m-%d-%Y")
# dir=${now}_${dataset}
# mkdir ${dir}

if [ $query_label -gt 0 ]; then
    if [ "$distribution" == "in_dist" ] || [ "$distribution" == "out_dist" ] ; then
        echo ""
    else 
        echo "err, only support in/out dist"
        exit 0
    fi
else 
    echo "err, query label is 0"
    exit 0
fi


if [ -f "$query_file" ]; then
    echo Query range file ${query_file} already exist. 
    exit 0
fi

# TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_sift_n=${N}.txt


python utils/BiasedLabelQueryGenerator.py ${query_size} \
                                        ${query_file} \
                                        ${query_range_file} \
                                        ${label_cnt} \
                                        ${label_range} \
                                        ${query_label_cnt} \
                                        ${distribution} \
                                        ${query_label} \
                                        ${centroid_file} \
                                        ${N} \
                                        ${dataset_attr_file} \
                                        ${ori_query_file} \
                                        ${ori_query_size}





