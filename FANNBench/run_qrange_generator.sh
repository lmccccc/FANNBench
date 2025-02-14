export debugSearchFlag=0
#! /bin/bash

now=$(date +"%m-%d-%Y")

source ./vars.sh construction $1 $2

# now=$(date +"%m-%d-%Y")
# dir=${now}_${dataset}
# mkdir ${dir}


if [ -f "$query_range_file" ]; then
    echo Query range file ${query_range_file} already exist. 
    exit 1
fi

# TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_sift_n=${N}.txt

python utils/qrangeGenerator.py ${query_size} \
                                ${query_file} \
                                ${query_range_file} \
                                ${label_cnt} \
                                ${label_range} \
                                ${query_label_cnt} \
                                ${distribution} \
                                ${query_label} \
                                ${centroid_file} \
                                ${query_label_sel}





