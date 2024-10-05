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
algo=acorn

# run of sift1M test

dir=${now}_${dataset}_${algo}
mkdir ${dir}
mkdir ${acorn_root}
mkdir ${acorn_index_root}

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${algo}_${dataset}.txt

echo "acorn index file: ${acorn_index_file}"

if [ -e $acorn_index_file ]; then
    echo "acorn index file already exist"
else
    echo  "construct index"
    ../ACORN/build/demos/acorn_build $dataset \
                                $N \
                                $gamma \
                                $M \
                                $M_beta \
                                $K \
                                $threads \
                                $dataset_file \
                                $dataset_attr_file \
                                $acorn_index_file \
                                &>> ${dir}/summary_${algo}_${dataset}.txt
fi

echo  "start query"
../ACORN/build/demos/acorn_query $dataset \
                                $N \
                                $gamma \
                                $M \
                                $M_beta \
                                $K \
                                $threads \
                                $dataset_file \
                                $query_file \
                                $dataset_attr_file \
                                $query_range_file \
                                $ground_truth_file \
                                $acorn_index_file \
                                $ef_search \
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

     





