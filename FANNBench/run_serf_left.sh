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

source ./vars.sh $1 $2 $3 $4
source ./file_check.sh
algo=SeRF_left

# run of sift1M test


#SeRF vars
index_store_path=${algo}_${dataset}_index.bin

dir=logs/${now}_${dataset}_${algo}


if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$serf_index_root" ]; then
    mkdir ${serf_root}
    mkdir ${serf_index_root}
fi

log_file=${dir}/summary_${algo}_${dataset}_efsearch${ef_search}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

echo "method: $algo"
echo "dataset: $dataset"
echo "N: $N"
echo "dataset_path: $dataset_file"
echo "query_path: $query_file"
echo "label_path: $dataset_attr_file"
echo "qrange_path: $query_range_file"
echo "groundtruth_path: $ground_truth_file"
echo "query_num: $query_size"
echo "K: $K"
echo "M: $M"
echo "ef_construction: $ef_construction"
echo "ef_search: $ef_search"
echo "index_file: $serf_index_file"
echo "nthread: $threads"
if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $serf_index_file ]; then
        echo "index file already exist"
        exit 0
    fi  
fi

/bin/time -v -p ../SeRF/build/benchmark/deep_arbitrary -method $algo \
                                       -dataset $dataset \
                                       -N $N \
                                       -dataset_path $dataset_file \
                                       -query_path $query_file \
                                       -label_path $dataset_attr_file \
                                       -qrange_path $query_range_file \
                                       -groundtruth_path $ground_truth_file \
                                       -query_num $query_size \
                                       -K $K \
                                       -M $serf_M \
                                       -ef_construction $ef_construction \
                                       -ef_search $ef_search \
                                       -index_file $serf_index_file \
                                       -ef_max $ef_max \
                                       -nthreads $threads \
                                       &>> $log_file

status=$?
if [ $status -ne 0 ]; then
    echo "serf failed to run."
else
    echo "serf succeed."
    source ./run_txt2csv.sh
fi