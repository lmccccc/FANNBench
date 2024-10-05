export debugSearchFlag=0
#! /bin/bash



# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

# make -C build -j faiss
# make -C build utils
# make -C build test_acorn



##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")


# run of sift1M test
N=1000000 
gamma=12 
dataset=sift1M_test 
M=32 
M_beta=64

dataset_file="../../dataset/sift/bigann_base.bvecs"
query_file="../../dataset/sift/bigann_query.bvecs"
dataset_attr_file="../../dataset/sift/sift_attr.json"
query_range_file="../../dataset/sift/sift_qrange.json"
ground_truth_file="../../dataset/sift/sift_gt_10.json"

parent_dir=${now}_${dataset}
mkdir ${parent_dir}
dir=${parent_dir}/MB${M_beta}
mkdir ${dir}

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt


./build/demos/test_acorn $N $gamma $dataset $M $M_beta $dataset_file $query_file $dataset_attr_file $query_range_file $ground_truth_file  &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt

     





