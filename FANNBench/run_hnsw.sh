# export debugSearchFlag=0
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
algo=HNSW

# run of sift1M test

dir=logs/${now}_${dataset}_${algo}

if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$hnsw_index_root" ]; then
    mkdir ${hnsw_root}
    mkdir ${hnsw_index_root}
fi

log_file=${dir}/summary_${algo}_${dataset}_efs${ef_search}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file


if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    echo "hnsw index file: ${hnsw_index_file}"
    if [ -e $hnsw_index_file ]; then
        echo "hnsw index file already exist"
        exit 0
    else
        echo  "construct index"
        /bin/time -v -p ../faiss/build/demos/hnsw_build $dataset \
                                    $N \
                                    $K \
                                    $threads \
                                    $dataset_file \
                                    $dataset_attr_file \
                                    $hnsw_index_file \
                                    $dim \
                                    $M \
                                    $ef_construction \
                                    &>> $log_file
        status=$?
        if [ $status -ne 0 ]; then
            echo "hnsw index failed with exit status $status"
        else
            echo "hnsw index ran successfully"
        fi
    fi
fi



if [ "$mode" == "query" ] || [ "$mode" == "all" ]; then
    if [ "$label_cnt" -eq 2 ]; then
        echo  "start arbitrary query"
        /bin/time -v -p ../faiss/build/demos/hnsw_query_arbi $dataset \
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
                                        &>> $log_file

    elif [ "$label_cnt" -gt 1 ]; then
        echo  "start keyword query"
        /bin/time -v -p ../faiss/build/demos/hnsw_query_keyword $dataset \
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
                                        &>> $log_file
    else
        echo  "start query"
        /bin/time -v -p ../faiss/build/demos/hnsw_query $dataset \
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
                                        &>> $log_file
    fi
    status=$?
    if [ $status -ne 0 ]; then
        echo "hnsw query failed with exit status $status"
    else
        echo "hnsw query ran successfully"
    fi
fi

     
if [ $status -eq 0 ]; then
    source ./run_txt2csv.sh
fi





