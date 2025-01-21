source ./vars.sh query


if [ "$1" == "ivfpq" ]; then
    echo "remove $ivfpq_index_file"
    rm $ivfpq_index_file
    exit 0
fi


if [ "$1" == "all" ]; then
    rm $dataset_bin_file
    rm $query_bin_file

    rm -rf $label_path
    rm -rf $acorn_root
    rm -rf $ivfpq_root
    rm -rf $hnsw_root
    rm -rf $diskann_root
    rm -rf $rfann_root
    rm -rf $vtree_root
    rm -rf $irange_root
    rm -rf $rii_root
    rm -rf $serf_root
    rm -rf $nhq_root
    rm -rf $nhqkg_root
    rm -rf $unify_root

    echo "remove $nhq_root"
    rm -rf $nhq_root
    exit 0
fi