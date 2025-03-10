echo "batch construct"

source ./vars.sh construction


run_func(){
    echo "Constructing index for $1"
    ./$1 $2
    status=$?
    if [ $status -ne 0 ]; then
        echo "script $1 failed with exit status $status"
        exit $status
    else
        echo "script $1 ran successfully"
    fi
}



echo "Construction mode"
# rm ${acorn_index_file}
# rm -rf ${diskann_index_label_root}
# rm ${hnsw_index_file}

# rm ${irange_index_file}
# rm ${irange_id2od_file}

# rm ${ivfpq_index_file}

# rm -rf ${nhq_index_model_file}
# rm -rf ${nhq_index_attr_file}

# rm -rf ${nhqkg_index_model_file}
# rm -rf ${nhqkg_index_attr_file}

# rm -rf ${rii_index_file}

# rm -rf ${serf_index_file}

# rm -rf ${vtree_index_prefix}

# rm -rf ${rfann_index_prefix}  # wst super optimized post filtering

# rm -rf ${unify_index_file}

# # remove all temporary attr files to ensure the correctness of the experiment
# rm $keyword_query_range_file
# rm $label_file
# rm $ground_truth_bin_file

# rm $query_bin_file
# rm $attr_bin_file

# rm $qrange_bin_file
# rm $attr_bin_file

# rm $dataset_attr_file
# rm $query_range_file
# rm $ground_truth_file
# rm $centroid_file
# rm $keyword_query_range_file
# rm $label_file
# rm $ground_truth_bin_file
# rm $attr_bin_file
# rm $qrange_bin_file
# rm $irange_id2od_file
# rm -rf $label_path

# echo "Generate attr"
# ./run_attr_generator.sh
# ./run_qrange_generator.sh
# ./run_groundtruth_generator.sh

# echo "Run all benchmarks"
if [ $1 == "range" ]; then
    echo "Range construction"
    # range 
    run_func run_acorn.sh construction          # Acorn
    run_func run_serf.sh construction           # SeRF
    run_func run_unify.sh construction          # UNIFY
    run_func run_milvus_ivfpq.sh construction   # Milvus IVFPQ
    run_func run_milvus_hnsw.sh construction    # Milvus HNSW
    run_func run_hnsw.sh construction           # Faiss HNSW
    run_func run_ivfpq.sh construction          # Faiss IVFPQ
    run_func run_irange.sh construction         # iRangeGraph
    run_func run_vamanatree.sh construction     # WST vamana tree
    run_func run_wst.sh construction            # WST super optimized post filtering
    run_func run_dsg.sh construction            # DSG
elif [ $1 == "keyword" ]; then
    echo "Keyword construction"
    ## keyword 
    run_func run_acorn.sh construction          # Acorn
    run_func run_hnsw.sh construction           # Faiss HNSW
    run_func run_ivfpq.sh construction          # Faiss IVFPQ
    run_func run_nhq_kgraph.sh construction     # NHQ-NPG KGraph
    run_func run_nhq_nsw.sh construction        # NHQ-NPG NSW
    run_func run_diskann.sh construction        # DiskANN
    run_func run_diskann_stitched.sh construction        # DiskANN_stitched
    run_func run_milvus_hnsw.sh construction           # Faiss HNSW
    run_func run_milvus_ivfpq.sh construction          # Faiss IVFPQ
else
    echo "Invalid construction mode(range or keyword)"
    exit 1
fi


# never used
# run_func run_rii.sh construction            # RII



