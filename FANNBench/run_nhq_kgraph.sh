export debugSearchFlag=0
#! /bin/bash
source ./vars.sh $1 $2 $3 $4
source ./file_check.sh

algo=NHQ_kgraph
##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")


dir=logs/${now}_${dataset}_${algo}

if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$nhqkg_index_root" ]; then
    mkdir ${nhqkg_root}
    mkdir ${nhqkg_index_root}
fi

# mkdir ${dir}

log_file=${dir}/summary_${algo}_${dataset}_efsearch${ef_search}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

if [ -e $keyword_query_range_file ]; then
    echo "query keyword txt file already exist at $keyword_query_range_file"
else
    echo "convert json range query label to keyword txt"
    python utils/range2keyword.py $query_range_file $keyword_query_range_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
        exit $status
    else
        echo "Python script ran successfully"
    fi
fi


if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $nhqkg_index_model_file ]; then
        echo "index file already exist at $nhqkg_index_model_file"
        exit 0
    else
        echo  "construct index"
        
        echo "dataset file: $dataset_file" 
        echo "label file: $dataset_attr_file"
        echo "save index model file: $nhqkg_index_model_file"
        echo "save index attr file: $nhqkg_index_attr_file"
        echo "K: $K"
        echo "L: $kgraph_L"
        echo "iter: $iter"
        echo "S: $S"
        echo "R: $R"
        echo "Range: $RANGE"
        echo "PL: $PL"
        echo "B: $B"
        echo "M: $kgraph_M"
        echo "data size: $N"
        echo "threads: $threads"
        
        /bin/time -v -p ../NHQ-main/NHQ-NPG_kgraph/build/tests/test_dng_index $dataset_file \
                                                            $dataset_attr_file \
                                                            $nhqkg_index_model_file \
                                                            $nhqkg_index_attr_file \
                                                            $K \
                                                            $kgraph_L \
                                                            $iter \
                                                            $S \
                                                            $R \
                                                            $RANGE \
                                                            $PL \
                                                            $B \
                                                            $kgraph_M \
                                                            $N \
                                                            $threads \
                                                            &>> $log_file

    #   std::cout << "data file: " << argv[1] << std::endl;
    #   std::cout << "label file: " << argv[2] << std::endl;
    #   std::cout << "save index model file: " << argv[3] << std::endl;
    #   std::cout << "save index attr file: " << argv[4] << std::endl;
    #   std::cout << "K: " << argv[5] << std::endl;
    #   std::cout << "L: " << argv[6] << std::endl;
    #   std::cout << "iter: " << argv[7] << std::endl;
    #   std::cout << "S: " << argv[8] << std::endl;
    #   std::cout << "R: " << argv[9] << std::endl;
    #   std::cout << "Range: " << argv[10] << std::endl;
    #   std::cout << "PL: " << argv[11] << std::endl;
    #   std::cout << "B: " << argv[12] << std::endl;
    #   std::cout << "M: " << argv[13] << std::endl;
    #   std::cout << "data size: " << argv[14] << std::endl;

        if [ $? -ne 0 ]; then
            echo "nhq kagraph index constructor failed to run."
        else
            echo "nhq kgraph index constructed."
        fi
    fi
fi

if [ "$mode" == "query" ] || [ "$mode" == "all" ]; then
    # --universal_label $universal_label \
    echo "index model file: $nhqkg_index_model_file" 
    echo "index attr file: $nhqkg_index_attr_file"
    echo "query file: $query_file"
    echo "data file: $data_file"
    echo "ground truth file: $ground_truth_file"
    echo "query label file: $keyword_query_range_file"
    echo "K: $K"
    echo "weight search: $weight_search"
    echo "L search: $L_search"
    /bin/time -v -p ../NHQ-main/NHQ-NPG_kgraph/build/tests/test_dng_optimized_search $nhqkg_index_model_file \
                                                                    $nhqkg_index_attr_file \
                                                                    $dataset_file \
                                                                    $query_file \
                                                                    $keyword_query_range_file \
                                                                    $ground_truth_file \
                                                                    $K \
                                                                    $weight_search \
                                                                    $L_search \
                                                                    &>> $log_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "nhq kgraph query failed with exit status $status"
    else
        echo "nhq kgraph query ran successfully"
    fi
fi


#   std::cout << "graph file: " << argv[1] << std::endl;
#   std::cout << "attributetable file: " << argv[2] << std::endl;
#   std::cout << "query file: " << argv[3] << std::endl;
#   std::cout << "data path: " << argv[4] << std::endl;
#   std::cout << "query attr file: " << argv[5] << std::endl;
#   std::cout << "groundtruth file: " << argv[6] << std::endl;
#   std::cout << "K: " << argv[7] << std::endl;
#   std::cout << "weight search: " << argv[8] << std::endl;
#   std::cout << "L search: " << argv[9] << std::endl;
status=$?
if [ $status -eq 0 ]; then
    source ./run_txt2csv.sh
fi