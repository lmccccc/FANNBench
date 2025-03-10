source vars.sh query



sel_list=(
    6   # 50%
    10  # 10%
    19  # 1%
    20  # 0.1%
)

# sel_list=(
#     50000   # 50%
#     10000  # 10%
#     1000  # 1%
#     100  # 0.1%
# )
rm_tmp_func(){
    rm $dataset_bin_file
    rm $query_bin_file
    rm $attr_bin_file
    rm $qrange_bin_file
    rm $ground_truth_bin_file
    rm $keyword_query_range_file
    rm $label_file
}

if [ $1 == "rmtmp" ]; then
    rm_tmp_func
    exit 0
fi

if [ $1 == "batch" ]; then
    if [ $2 == "range" ]; then
        python utils/modify_var.py label_range 100000
        python utils/modify_var.py label_cnt 1
        python utils/modify_var.py query_label_cnt 6
        python utils/modify_var.py query_label 0
        if [ ! -f $dataset_attr_file ]; then
            echo Attribute file ${dataset_attr_file} not exist. 
        fi
        if [ ! -e $acorn_index_file ]; then
            echo "acorn index file ${acorn_index_file} not exist"
        fi
        if [ ! -e $ivfpq_index_file ]; then
            echo "ivfpq index file ${ivfpq_index_file} not exist"
        fi
        if [ ! -e $hnsw_index_file ]; then
            echo "hnsw index file ${hnsw_index_file} not exist"
        fi
        if [ ! -d $rfann_index_prefix ]; then
            echo "WST opt index file ${rfann_index_prefix} not exist"
        fi
        if [ ! -d $vtree_index_prefix ]; then
            echo "WST vtree index file ${vtree_index_prefix} not exist"
        fi
        if [ ! -e $irange_index_file ]; then
            echo "irange index file ${irange_index_file} not exist"
        fi
        if [ ! -e $irange_id2od_file ]; then
            echo "irange order file ${irange_id2od_file} not exist"
        fi
        if [ ! -e $serf_index_file ]; then
            echo "serf index file ${serf_index_file} not exist"
        fi
        if [ ! -e $dsg_index_file ]; then
            echo "dsg index file ${dsg_index_file} not exist"
        fi
        if [ ! -e $unify_index_file ]; then
            echo "unify index file ${unify_index_file} not exist"
        fi

        python utils/check_collection_exist.py $hnsw_collection_name
        python utils/check_collection_exist.py $collection_name

        for qrange in "${sel_list[@]}"; do
            echo "qrange=$qrange"
            python utils/modify_var.py query_label_cnt $qrange
            if [ ! -f "$query_range_file" ]; then
                echo qrange file ${query_range_file} not exist. 
            fi
            if [ ! -f "$ground_truth_file" ]; then
                echo ground truth file ${ground_truth_file} not exist. 
            fi

        done
        
    elif [ $2 == "keyword" ]; then
        python utils/modify_var.py label_range 500
        python utils/modify_var.py label_cnt 1
        python utils/modify_var.py query_label_cnt 1
        python utils/modify_var.py query_label 6
        
            
        if [ ! -e $acorn_index_file ]; then
            echo "acorn index file ${acorn_index_file} not exist"
        fi
        if [ ! -e $diskann_index_file ]; then
            echo "diskann index file ${diskann_index_file} not exist"
        fi
        if [ ! -e $diskann_stit_index_file ]; then
            echo "diskann stitched index file ${diskann_stit_index_file} not exist"
        fi
        if [ ! -e $hnsw_index_file ]; then
            echo "hnsw index file ${hnsw_index_file} not exist"
        fi
        if [ ! -e $ivfpq_index_file ]; then
            echo "ivfpq index file ${ivfpq_index_file} not exist"
        fi
        if [ ! -e $nhqkg_index_model_file ]; then
            echo "nhq kgraph index file ${nhqkg_index_model_file} not exist"
        fi
        if [ ! -e $nhq_index_model_file ]; then
            echo "nhq nsw index file ${nhq_index_model_file} not exist"
        fi
        python utils/check_collection_exist.py $hnsw_collection_name
        python utils/check_collection_exist.py $collection_name

        if [ ! -f "$dataset_attr_file" ]; then
            echo Attribute file ${dataset_attr_file} not exist. 
        fi
        for q_label in "${sel_list[@]}"; do
            python utils/modify_var.py query_label $query_label
            if [ ! -f "$query_range_file" ]; then
                echo qrange file ${query_range_file} not exist. 
            fi
            if [ ! -f "$ground_truth_file" ]; then
                echo ground truth file ${ground_truth_file} not exist. 
            fi

        done
    fi
    exit 0
fi