export debugSearchFlag=0
#! /bin/bash



# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

# make -C build -j faiss
# make -C build utils
# make -C build generate_groundtruth



##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")


# run of sift1M test
# dataset=sift1M
# N=1000000 
# K=10
# dataset_file=../ACORN/Datasets/sift1M/sift_base.fvecs
# query_file=../ACORN/Datasets/sift1M/sift_query.fvecs
# attr_file=../ACORN/testing_data/sift_attr.ivecs
# qrange_file=../ACORN/testing_data/sift_qrange.ivecs
# output_file=../ACORN/testing_data/sift_gt.txt   #stored like ivecs, but long int in faiss

#sift1B
dataset=sift1B
N=1000000000 
K=100
dataset_file="../../dataset/sift/bigann_base.bvecs"
query_file="../../dataset/sift/bigann_query.bvecs"
attr_file="../../dataset/sift/sift_attr.json"
qrange_file="../../dataset/sift/sift_qrange.json"
output_file="../../dataset/sift/sift_qrange.ivecs"   #stored like ivecs, but long int in faiss


parent_dir=${now}_${dataset}
mkdir ${parent_dir}
dir=logs
mkdir ${parent_dir}/${dir}

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${parent_dir}/${dir}/generate_gt_log.txt

# <number vecs> <dataset> <attr> <query> <queryrange> <output> <k>
./build/demos/generate_groundtruth $N $dataset_file $attr_file $query_file $qrange_file $output_file $K  &>> ${parent_dir}/${dir}/summary_sift_n=${N}.txt