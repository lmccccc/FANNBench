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

source ./vars.sh $1 $2 $3
source ./file_check.sh
algo=DSG

# run of sift1M test


#dsg vars
index_store_path=${algo}_${dataset}_index.bin

dir=logs/${now}_${dataset}_${algo}


if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$dsg_index_root" ]; then
    mkdir ${dsg_root}
    mkdir ${dsg_index_root}
fi

log_file=${dir}/summary_${algo}_${dataset}_efsearch${ef_search}.txt
# TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

# echo "method: $algo"
# echo "dataset: $dataset"
# echo "N: $N"
# echo "dataset_path: $dataset_file"
# echo "query_path: $query_file"
# echo "label_path: $dataset_attr_file"
# echo "qrange_path: $query_range_file"
# echo "groundtruth_path: $ground_truth_file"
# echo "query_num: $query_size"
# echo "K: $K"
# echo "M: $M"
# echo "ef_construction: $ef_construction"
# echo "ef_search: $ef_search"
# echo "index_file: $dsg_index_file"
# echo "nthread: $threads"
if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $dsg_index_file ]; then
        echo "index file already exist at $dsg_index_file"
        exit 1
    else
        echo "construct index"
        echo "dataset: $dataset"
        echo "N: $N"
        echo "dataset_path: $dataset_file"
        echo "index_path: $dsg_index_file"
        echo "method: compact"
        echo "k: $K"
        echo "ef_max: $ef_max"
        echo "ef_construction: $ef_construction"
        echo "label_path: $dataset_attr_file"
        echo "nthreads: $threads"
        # echo "../DynamicSegmentGraph/build/benchmark/build_index -dataset $dataset -N $N -dataset_path $dataset_file -index_path $dsg_index_file -method "compact" -k $K -ef_max $ef_max -ef_construction $ef_construction -label_path $dataset_attr_file -nthreads $threads -query_path $query_file &>> $log_file"
        /bin/time -v -p ../DynamicSegmentGraph/build/benchmark/build_index -dataset $dataset \
                                                                            -N $N \
                                                                            -dataset_path $dataset_file \
                                                                            -index_path $dsg_index_file \
                                                                            -method "compact" \
                                                                            -k $serf_M \
                                                                            -ef_max $ef_max \
                                                                            -ef_construction $ef_construction \
                                                                            -label_path $dataset_attr_file \
                                                                            -nthreads $threads \
                                                                            -query_path $query_file \
                                                                            &>> $log_file
        status=$?
        if [ $status -ne 0 ]; then
            echo "dsg constructor failed to run."
        else
            echo "dsg index constructed."
        fi
    fi
fi

if [ "$mode" == "query" ] || [ "$mode" == "all" ]; then

    /bin/time -v -p ../DynamicSegmentGraph/build/benchmark/query_index -dataset $dataset \
                                                                        -N $N \
                                                                        -dataset_path $dataset_file \
                                                                        -query_path $query_file \
                                                                        -groundtruth_path $ground_truth_file \
                                                                        -index_path $dsg_index_file \
                                                                        -method "compact" \
                                                                        -index_path $dsg_index_file \
                                                                        -k $serf_M \
                                                                        -ef_max $ef_max \
                                                                        -ef_construction $ef_construction \
                                                                        -nthreads $threads \
                                                                        -ef_search $ef_search \
                                                                        -qrange_path $query_range_file \
                                                                        -query_num $query_size \
                                                                        -label_path $dataset_attr_file \
                                                                        -query_k $K \
                                                                        &>> $log_file
    status=$?
    if [ $status -ne 0 ]; then
        echo "diskann query failed with exit status $status"
    else
        echo "diskann query ran successfully"
    fi
fi

if [ $status -ne 0 ]; then
    echo "dsg failed to run."
else
    echo "dsg succeed."
fi

if [ $status -eq 0 ]; then
    source ./run_txt2csv.sh
fi