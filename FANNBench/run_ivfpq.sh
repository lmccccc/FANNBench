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
algo=IVFPQ

# run of sift1M test

dir=logs/${now}_${dataset}_${algo}

log_file=${dir}/summary_${algo}_${dataset}_nprobe${nprobe}.txt

if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$ivfpq_index_root" ]; then
    mkdir ${ivfpq_root}
    mkdir ${ivfpq_index_root}
fi

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

echo "ivfpq index file: ${ivfpq_index_file}"

if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $ivfpq_index_file ]; then
        echo "ivfpq index file already exist"
        exit 1
    else
        echo  "construct index"
        /bin/time -v -p ../faiss/build/demos/ivfpq_build $dataset \
                                    $N \
                                    $K \
                                    $threads \
                                    $dataset_file \
                                    $dataset_attr_file \
                                    $ivfpq_index_file \
                                    $dim \
                                    $partition_size_M \
                                    $train_file \
                                    &>> $log_file

        status=$?
        if [ $status -ne 0 ]; then
            echo "ivfpq index failed with exit status $status"
        else
            echo "ivfpq index ran successfully"
        fi
    fi
fi

if [ "$mode" == "query" ] || [ "$mode" == "all" ]; then
    if [ "$label_cnt" -gt 1 ]; then
        echo  "start keyword query"
        /bin/time -v -p ../faiss/build/demos/ivfpq_query_keyword $dataset \
                                        $N \
                                        $K \
                                        $threads \
                                        $dataset_file \
                                        $query_file \
                                        $dataset_attr_file \
                                        $query_range_file \
                                        $ground_truth_file \
                                        $ivfpq_index_file \
                                        $nprobe \
                                        $dim \
                                        &>> $log_file

    else
        echo  "start query"
        /bin/time -v -p ../faiss/build/demos/ivfpq_query $dataset \
                                        $N \
                                        $K \
                                        $threads \
                                        $dataset_file \
                                        $query_file \
                                        $dataset_attr_file \
                                        $query_range_file \
                                        $ground_truth_file \
                                        $ivfpq_index_file \
                                        $nprobe \
                                        $dim \
                                        &>> $log_file
    fi

    status=$?
    if [ $status -ne 0 ]; then
        echo "ivfpq query failed with exit status $status"
    else
        echo "ivfpq quey ran successfully"
    fi    
fi

if [ $status -eq 0 ]; then
    source ./run_txt2csv.sh
fi





