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

source ./vars.sh
source ./file_check.sh
algo=HNSW

# run of sift1M test

dir=logs/${now}_${dataset}_${algo}
mkdir ${dir}
mkdir ${hnsw_root}
mkdir ${hnsw_index_root}

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${algo}_${dataset}.txt

echo "hnsw index file: ${hnsw_index_file}"

if [ -e $hnsw_index_file ]; then
    echo "hnsw index file already exist"
else
    echo  "construct index"
    ../faiss/build/demos/hnsw_build $dataset \
                                $N \
                                $K \
                                $threads \
                                $dataset_file \
                                $dataset_attr_file \
                                $hnsw_index_file \
                                $dim \
                                $M \
                                $ef_construction \
                                &>> ${dir}/summary_${algo}_${dataset}.txt
fi

echo  "start query"
../faiss/build/demos/hnsw_query $dataset \
                                $N \
                                $K \
                                $threads \
                                $dataset_file \
                                $query_file \
                                $dataset_attr_file \
                                $query_range_file \
                                $ground_truth_file \
                                $hnsw_index_file \
                                $ef_search \
                                $dim \
                                &>> ${dir}/summary_${algo}_${dataset}.txt

# ../ACORN/build/demos/test_acorn $N \
#                                 $gamma \
#                                 $dataset \
#                                 $M \
#                                 $M_beta \
#                                 $dataset_file \
#                                 $query_file \
#                                 $dataset_attr_file \
#                                 $query_range_file \
#                                 $ground_truth_file \
#                                 $gt_topk \
#                                 $acorn_result_root \
#                                 &>> ${dir}/summary_${algo}_${dataset}.txt

     





