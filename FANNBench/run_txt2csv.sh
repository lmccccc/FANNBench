# echo "dir: $dir"
# echo "dataset: $dataset"
# echo "label_method: $label_method"
# echo "query_method: $query_method"
# echo "distribution: $distribution"
# echo "label_range: $label_range"
# echo "query_label_cnt: $query_label_cnt"
# echo "ef_construction: $ef_construction"
# echo "M: $M"
# echo "serf_M: $serf_M"
# echo "K: $K"
# echo "nprobe: $nprobe"
# echo "ef_search: $ef_search"
# echo "threads: $threads"
# echo "gamma: $gamma"
# echo "M_beta: $M_beta"
# echo "alpha: $alpha"
# echo "L: $L"
# echo "partition_size_M: $partition_size_M"
# echo "beamsize: $beamsize"
# echo "split_factor: $split_factor"
# echo "shift_factor: $shift_factor"
# echo "final_beam_multiply: $final_beam_multiply"
# echo "kgraph_L: $kgraph_L"
# echo "iter: $iter"
# echo "S: $S"
# echo "R: $R"
# echo "B: $B"
# echo "kgraph_M: $kgraph_M"
# echo "weight_search: $weight_search"
# echo "dim: $dim"
# echo "N: $N"
# echo "query_size: $query_size"
# echo "acorn_index_file: $acorn_index_file"
# echo "diskann_index_label_root: $diskann_index_label_root"
# echo "rfann_index_prefix: $rfann_index_prefix"
# echo "irange_index_file: $irange_index_file"
# echo "rii_index_file: $rii_index_file"
# echo "serf_index_file: $serf_index_file"
# echo "nhq_index_model_file: $nhq_index_model_file"
# echo "nhq_index_attr_file: $nhq_index_attr_file"
# echo "ivfpq_index_file: $ivfpq_index_file"
# echo "hnsw_index_file: $hnsw_index_file"
# echo "vtree_index_prefix: $vtree_index_prefix"
# echo "nhqkg_index_model_file: $nhqkg_index_model_file"
# echo "nhqkg_index_attr_file: $nhqkg_index_attr_file"
# echo "milvus_coll_path: $milvus_coll_path"
# echo "algo: $algo"
# echo "result_file: $result_file"
# echo "mode: $mode"
# echo "B_unify: $B_unify"
# echo "AL: $AL"
# echo "unify_index_file: $unify_index_file"
# echo "Stitched_R: $Stitched_R"
# echo "ef_max: $ef_max"
# echo "dsg_index_file: $dsg_index_file"


python -u utils/extract_results.py $log_file \
                                   $dataset \
                                   $label_cnt \
                                   $query_label \
                                   $distribution \
                                   $label_range \
                                   $query_label_cnt \
                                   $ef_construction \
                                   $M \
                                   $serf_M \
                                   $K \
                                   $nprobe \
                                   $ef_search \
                                   $threads \
                                   $gamma \
                                   $M_beta \
                                   $alpha \
                                   $L \
                                   $partition_size_M \
                                   $beamsize \
                                   $split_factor \
                                   $shift_factor \
                                   $final_beam_multiply \
                                   $kgraph_L \
                                   $iter \
                                   $S \
                                   $R \
                                   $B \
                                   $kgraph_M \
                                   $weight_search \
                                   $dim \
                                   $N \
                                   $query_size \
                                   $acorn_index_file \
                                   $diskann_index_label_root \
                                   $rfann_index_prefix \
                                   $irange_index_file \
                                   $rii_index_file \
                                   $serf_index_file \
                                   $nhq_index_model_file \
                                   $nhq_index_attr_file \
                                   $ivfpq_index_file \
                                   $hnsw_index_file \
                                   $vtree_index_prefix \
                                   $nhqkg_index_model_file \
                                   $nhqkg_index_attr_file \
                                   $milvus_coll_path \
                                   $algo \
                                   $result_file \
                                   $mode \
                                   $B_unify \
                                   $AL \
                                   $unify_index_file \
                                   $Stitched_R \
                                   $ef_max \
                                   $dsg_index_file