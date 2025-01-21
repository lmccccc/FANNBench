echo "batch construct"

source ./vars.sh construction $2 $3


run_func(){
    echo "Run $1"
    ./$1 $2
    status=$?
    if [ $status -ne 0 ]; then
        echo "script $1 failed with exit status $status"
        exit $status
    else
        echo "script $1 ran successfully"
    fi
}



# echo "Construction mode, remove all index files"
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

## range query
run_func run_acorn.sh construction $2 $3          # Acorn
run_func run_serf.sh construction $2 $3           # SeRF
run_func run_unify.sh construction $2 $3          # UNIFY
run_func run_milvus_ivfpq.sh construction $2 $3   # Milvus IVFPQ
run_func run_milvus_hnsw.sh construction $2 $3    # Milvus HNSW
run_func run_hnsw.sh construction $2 $3           # Faiss HNSW
run_func run_ivfpq.sh construction $2 $3          # Faiss IVFPQ
run_func run_irange.sh construction $2 $3         # iRangeGraph
run_func run_vamanatree.sh construction $2 $3     # WST vamana tree
run_func run_wst.sh construction $2 $3            # WST super optimized post filtering
run_func run_dsg.sh construction $2 $3            # DSG

## keyword query
# run_func run_nhq_kgraph.sh construction $2 $3     # NHQ-NPG KGraph
# run_func run_nhq_nsw.sh construction $2 $3        # NHQ-NPG NSW
# run_func run_diskann.sh construction $2 $3        # DiskANN
# run_func run_diskann_stitched.sh construction $2 $3        # DiskANN_stitched


# never used
# run_func run_rii.sh construction $2 $3            # RII



