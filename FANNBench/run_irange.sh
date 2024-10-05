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
algo=iRangeGraph

# run of sift1M test

dir=${now}_${dataset}_${algo}
mkdir ${dir}
mkdir ${irange_root}
mkdir ${irange_index_root}
mkdir ${irange_result_root}

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${algo}_${dataset}.txt

if [ -e $dataset_bin_file ]; then
    echo "dataset bin already exist"
else
    echo "convert base vecs to bin"
    # same file format used with DiskANN, so use it
    ./../DiskANN/build/apps/utils/fvecs_to_bin float $dataset_file $dataset_bin_file
fi

if [ -e $query_bin_file ]; then
    echo "query bin already exist"
else
    echo "convert query vecs to bin"
    # same file format used with DiskANN, so use it
    ./../DiskANN/build/apps/utils/fvecs_to_bin float $query_file $query_bin_file
fi

if [ -e $attr_bin_file ]; then
    echo "query range bin file already exist"
else
    echo "convert json attr to bin, N inetger."
    python utils/qrange_json2bin.py $dataset_attr_file $attr_bin_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
        exit $status
    else
        echo "Python script ran successfully"
    fi
fi


if [ -e $irange_index_file ]; then
    echo "index exist"
else
    echo "construct index"
    echo "N: $N"
    echo "data_path: $dataset_bin_file"
    echo "index_file: $irange_index_file"
    echo "attr_file: $attr_bin_file"
    echo "id2od_file: $irange_id2od_file"
    echo "M: $M"
    echo "ef_construction: $ef_construction"
    echo "threads: $threads"
    ./../iRangeGraph/build/tests/buildindex --N $N \
                                            --data_path $dataset_bin_file \
                                            --index_file $irange_index_file \
                                            --attr_file $attr_bin_file \
                                            --id2od_file $irange_id2od_file  \
                                            --M $M \
                                            --ef_construction $ef_construction \
                                            --threads $threads \
                                            &>> ${dir}/summary_${algo}_${dataset}.txt
fi


if [ -e $qrange_bin_file ]; then
    echo "query range bin file already exist"
else
    echo "convert json query range to bin, N*2 inetger formated as left, right, left, right..."
    python utils/qrange_json2bin.py $query_range_file $qrange_bin_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
        exit $status
    else
        echo "Python script ran successfully"
    fi
fi

if [ -e $ground_truth_bin_file ]; then
    echo "groundtruth bin file already exist"
else
    echo "convert json range query label to keyword txt"
    python utils/gt_json2bin.py $ground_truth_file $ground_truth_bin_file $gt_topk

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
        exit $status
    else
        echo "Python script ran successfully"
    fi
fi

#data_path, index_file, M, ef_construction, threads
echo "dataset bin file: $dataset_bin_file"
echo "query bin file: $query_bin_file"
echo "attr bin file: $attr_bin_file"
echo "qrange bin file: $qrange_bin_file"
echo "ground truth bin file: $ground_truth_bin_file"
echo "index file: $irange_index_file"
echo "result file: $irange_result_file"
echo "id2od file: $irange_id2od_file"
echo "N: $N"
echo "M: $M"
echo "Nq: $query_size"
echo "K: $K"
../iRangeGraph/build/tests/search --data_path $dataset_bin_file\
                                  --query_path $query_bin_file \
                                  --attr_file $attr_bin_file \
                                  --range_saveprefix $qrange_bin_file \
                                  --groundtruth_saveprefix $ground_truth_bin_file \
                                  --index_file $irange_index_file \
                                  --result_saveprefix $irange_result_file \
                                  --M $M \
                                  --id2od_file $irange_id2od_file \
                                  --N $N \
                                  --Nq $query_size \
                                  --K $K \
                                  --ef_search $ef_search \
                                  &>> ${dir}/summary_${algo}_${dataset}.txt

     





