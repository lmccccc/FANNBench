echo "mode: $1"

source ./vars.sh $1



if [ "$mode" == "construction" ]; then
    echo "Construction mode, remove all index files"
    rm ${acorn_index_file}
    rm -rf ${diskann_index_label_root}
    rm ${hnsw_index_file}

    rm ${irange_index_file}
    rm ${irange_id2od_file}

    rm ${ivfpq_index_file}

    rm -rf ${nhq_index_model_file}
    rm -rf ${nhq_index_attr_file}

    rm -rf ${nhqkg_index_model_file}
    rm -rf ${nhqkg_index_attr_file}

    rm -rf ${rii_index_file}

    rm -rf ${serf_index_file}

    rm -rf ${vtree_index_prefix}

    rm -rf ${rfann_index_prefix}  # wst super optimized post filtering

    # remove all temporary attr files to ensure the correctness of the experiment
    rm $keyword_query_range_file
    rm $label_file
    rm $ground_truth_bin_file

    rm $query_bin_file
    rm $attr_bin_file

    rm $qrange_bin_file
    rm $attr_bin_file
fi


run_func(){
    echo "Run $1"
    ./$1 $2
    status=$?
    if [ $status -ne 0 ]; then
        echo "script $1 failed with exit status $status"
        exit $status
    else
        echo "script ran successfully"
    fi
}

echo "Run all benchmarks"
run_func run_acorn.sh $1          # Acorn
run_func run_diskann.sh $1        # DiskANN
run_func run_hnsw.sh $1           # Faiss HNSW
run_func run_irange.sh $1         # iRangeGraph
run_func run_ivfpq.sh $1          # Faiss IVFPQ
run_func run_milvus_ivfpq.sh $1   # Milvus IVFPQ
run_func run_milvus_hnsw.sh $1   # Milvus HNSW
run_func run_nhq_kgraph.sh $1     # NHQ-NPG KGraph
run_func run_nhq_nsw.sh $1        # NHQ-NPG NSW
run_func run_rii.sh $1            # RII
run_func run_serf.sh $1           # SeRF
run_func run_vamanatree.sh $1     # WST vamana tree
run_func run_wst.sh $1            # WST super optimized post filtering
