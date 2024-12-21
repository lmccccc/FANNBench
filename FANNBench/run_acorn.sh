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
algo=ACORN

# run of sift1M test

dir=logs/${now}_${dataset}_${algo}

if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$acorn_index_root" ]; then
    mkdir ${acorn_root}
    mkdir ${acorn_index_root}
fi

log_file=${dir}/summary_${algo}_${dataset}_efs${ef_search}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

echo "acorn index file: ${acorn_index_file}"

if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $acorn_index_file ]; then
        echo "acorn index file already exist"
    else
        echo  "construct index"
        /bin/time -v -p ../ACORN/build/demos/acorn_build $dataset \
                                    $N \
                                    $gamma \
                                    $M \
                                    $M_beta \
                                    $K \
                                    $threads \
                                    $dataset_file \
                                    $dataset_attr_file \
                                    $acorn_index_file \
                                    $dim \
                                    &>> $log_file
        status=$?
        if [ $status -ne 0 ]; then
            echo "acorn index failed with exit status $status"
            exit $status
        else
            echo "acorn index ran successfully"
        fi
    fi
fi

if [ "$mode" == "query" ] || [ "$mode" == "all" ]; then

    if [ "$label_cnt" -gt 1 ]; then
        echo  "start query, efs: $ef_search"
        /bin/time -v -p ../ACORN/build/demos/acorn_query_keyword $dataset \
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
                                        $dim \
                                        &>> $log_file
    else
        echo  "start query keyword, efs: $ef_search"
        /bin/time -v -p ../ACORN/build/demos/acorn_query $dataset \
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
                                        $dim \
                                        &>> $log_file
    fi

    status=$?
    if [ $status -ne 0 ]; then
        echo "acorn query failed with exit status $status"
        exit $status
    else
        echo "acorn query ran successfully"
    fi
fi

     

source ./run_txt2csv.sh
