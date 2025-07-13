run_algo=$1

# search var list 
# acorn hnsw irange nhq_nsw serf dsg unify
ef_search_list=(
    # 3
    # 5
    # 8
    # 9
    # 10
    # 12
    # 15
    # 18
    # 20
    # 25
    30
    35
    40
    60
    80
    100
    120
    150
    180
    200
    300
    # 400
    # 500
    # 600
    # 700
    # 800
    # 900
    # 1000
    # 1200
    # 1400
    # 1600
)

# diskann diskann_stitched 
L_list=(
    # 5
    # 8
    # 10
    # 12
    # 15
    # 20
    # 40
    # 60
    # 80
    100
    150
    200
    300
    400
    500
    600
    700
    800
    900
    1000
    1200
    1400
)

nprobe_list=( # ivfpq
    # 1
    # 3
    # 5
    10
    20
    30
    50
    80
    100
    120
    150
    200
    300
    # 400
    # 500
    # 600
    # 700
    # 800
    # 900
    # 1000
)

# nhq_kgraph
L_search_list=( 
    10
    13
    15
    18
    20
    40
    60
    80
    100
    150
    200
    300
    400
    500
    1000
    # 1200
    # 1400
    # 1600
)

# wst_vamana wst_opt
beamsize_list=(
    # 2
    # 3
    # 5
    10
    # 12
    15
    # 18
    20
    40
    60
    80
    100
    150
    200
    300
    # 400
    # 500
    # 600
    # 700
    # 800
    # 900
    # 1000
)

# wst_opt
final_beam_multiply_list=(
    2
)

#unify
al_list=(
    # 1
    # 2
    4
    8
    16
    32
)

qrange_list=(
    100   #  0.1%
    1000  #  1 %
    # 2000
    # 3000
    # 4000
    # 5000
    # 6000
    # 7000
    # 8000
    # 9000
    10000 # 10 %
    50000 # 50 %
)

sel_list=(
    # 1   # 100%
    # 2   # 90%
    # 3   # 80%
    # 4   # 70%
    # 5   # 60%
    6   # 50%
    # 7   # 40%
    # 8   # 30%
    # 9   # 20%
    10  # 10%
    # 11  # 9%
    # 12  # 8%
    # 13  # 7%
    # 14  # 6%
    # 15  # 5%
    # 16  # 4%
    # 17  # 3%
    # 18  # 2%
    19  # 1%
    20  # 0.1%
)

# sel_list=(
#     # 100000
#     # 90000
#     # 80000
#     # 70000
#     # 60000
#     50000   # 50%
#     # 40000
#     # 30000
#     # 20000
#     10000  # 10%
#     # 9000
#     # 8000
#     # 7000
#     # 6000
#     # 5000
#     # 4000
#     # 3000
#     # 2000
#     1000  # 1%
#     100  # 0.1%
# )


if [ "$mode" == "construction" ]; then
    echo "use run_all.sh to construct"
    exit
fi


# run_func(){
#     echo "Run $1 $2 $3 $4"
#     ./$1 $2 $3 $4 &
#     status=$?
#     if [ $status -ne 0 ]; then
#         echo "script $1 failed with exit status $status"
#         exit $status
#     else
#         echo "script ran successfully"
#     fi
# }

query_func(){
    run_algo=$1
    echo "Run $run_algo"
    if [ "$run_algo" == "acorn" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search acorn at efs=$efs"
                source run_acorn.sh query multi_hnsw $efs $2          # Acorn
            else
                echo "search acorn at efs=$efs"
                source run_acorn.sh query multi_hnsw $efs $2 &          # Acorn
            fi
        done
    elif [ "$run_algo" == "acorn_rng" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search acorn at efs=$efs"
                source run_acorn_rng.sh query multi_hnsw $efs $2          # Acorn
            else
                echo "search acorn at efs=$efs"
                source run_acorn_rng.sh query multi_hnsw $efs $2 &          # Acorn
            fi
        done
    elif [ "$run_algo" == "acorn_kg" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search acorn at efs=$efs"
                source run_acorn_kg.sh query multi_hnsw $efs $2          # Acorn
            else
                echo "search acorn at efs=$efs"
                source run_acorn_kg.sh query multi_hnsw $efs $2 &          # Acorn
            fi
        done

    elif [ "$run_algo" == "diskann" ]; then
        for L in "${L_list[@]}"; do
            last_value=${L_list[-1]}
            if [ $L == $last_value ]; then
                echo "search diskann at L=$L"
                source run_diskann.sh query multi_diskann $L $2          # DiskANN
            else
                echo "search diskann at L=$L"
                source run_diskann.sh query multi_diskann $L $2 &          # DiskANN
            fi    
        done

    elif [ "$run_algo" == "diskann_stitched" ]; then
        for L in "${L_list[@]}"; do
            last_value=${L_list[-1]}
            if [ $L == $last_value ]; then
                echo "search diskann at L=$L"
                source run_diskann_stitched.sh query multi_diskann $L $2          # DiskANN
            else
                echo "search diskann at L=$L"
                source run_diskann_stitched.sh query multi_diskann $L $2 &          # DiskANN
            fi      
        done

    elif [ "$run_algo" == "hnsw" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search hnsw at efs=$efs"
                source run_hnsw.sh query multi_hnsw $efs $2          # Faiss HNSW
            else
                echo "search hnsw at efs=$efs"
                source run_hnsw.sh query multi_hnsw $efs $2 &          # Faiss HNSW
            fi
        done

    elif [ "$run_algo" == "hnsw_kg" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search hnsw at efs=$efs"
                source run_hnsw_kg.sh query multi_hnsw $efs $2          # Faiss HNSW
            else
                echo "search hnsw at efs=$efs"
                source run_hnsw_kg.sh query multi_hnsw $efs $2 &          # Faiss HNSW
            fi
        done


    elif [ "$run_algo" == "hnsw_2hop" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search hnsw at efs=$efs"
                source run_hnsw_2hop.sh query multi_hnsw $efs $2          # Faiss HNSW
            else
                echo "search hnsw at efs=$efs"
                source run_hnsw_2hop.sh query multi_hnsw $efs $2 &          # Faiss HNSW
            fi
        done

    elif [ "$run_algo" == "irange" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search irange at efs=$efs"
                source run_irange.sh query multi_hnsw $efs $2          # iRangeGraph
            else
                echo "search irange at efs=$efs"
                source run_irange.sh query multi_hnsw $efs $2 &          # iRangeGraph
            fi
        done

    elif [ "$run_algo" == "ivfpq" ]; then
        for nprobe in "${nprobe_list[@]}"; do
            last_value=${nprobe_list[-1]}
            if [ $nprobe == $last_value ]; then
                echo "search ivfpq at nprobe=$nprobe"
                source run_ivfpq.sh query multi_ivfpq $nprobe $2          # Faiss IVFPQ
            else
                echo "search ivfpq at nprobe=$nprobe"
                source run_ivfpq.sh query multi_ivfpq $nprobe $2 &          # Faiss IVFPQ
            fi
        done

    elif [ "$run_algo" == "milvus_ivfpq" ]; then 
        for nprobe in "${nprobe_list[@]}"; do
            last_value=${nprobe_list[-1]}
            if [ $nprobe == $last_value ]; then
                echo "search milvus ivfpq at nprobe=$nprobe"
                source run_milvus_ivfpq.sh query multi_ivfpq $nprobe $2 # since milvus connects to server, parallel run may cause error
            else
                echo "search milvus ivfpq at nprobe=$nprobe"
                source run_milvus_ivfpq.sh query multi_ivfpq $nprobe $2 & # since milvus connects to server, parallel run may cause error
            fi
        done

    elif [ "$run_algo" == "milvus_hnsw" ]; then 
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search milvus hnsw at efs=$efs"
                source run_milvus_hnsw.sh query multi_hnsw $efs $2 # since milvus connects to server, parallel run may cause error
            else
                echo "search milvus hnsw at efs=$efs"
                source run_milvus_hnsw.sh query multi_hnsw $efs $2 & # since milvus connects to server, parallel run may cause error
            fi
        done

    elif [ "$run_algo" == "kgraph" ]; then
        for l_search in "${L_search_list[@]}"; do
            last_value=${L_search_list[-1]}
            if [ $l_search == $last_value ]; then
                echo "search nhq kgraph at L_search=$l_search"
                source run_nhq_kgraph.sh query multi_kgraph $l_search $2          # NHQ-NPG KGraph
            else
                echo "search nhq kgraph at L_search=$l_search"
                source run_nhq_kgraph.sh query multi_kgraph $l_search $2 &          # NHQ-NPG KGraph
            fi
        done

    elif [ "$run_algo" == "nsw" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search nhq nsw at efs=$efs"
                source run_nhq_nsw.sh query multi_hnsw $efs $2          # NHQ-NPG NSW
            else
                echo "search nhq nsw at efs=$efs"
                source run_nhq_nsw.sh query multi_hnsw $efs $2 &          # NHQ-NPG NSW
            fi
        done

    elif [ "$run_algo" == "rii" ]; then
        source run_rii.sh query $2

    elif [ "$run_algo" == "serf" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search serf at efs=$efs"
                source run_serf.sh query multi_hnsw $efs $2          # SeRF
            else
                echo "search serf at efs=$efs"
                source run_serf.sh query multi_hnsw $efs $2 &          # SeRF
            fi
        done

    elif [ "$run_algo" == "serf_kg" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search serf at efs=$efs"
                source run_serf_kg.sh query multi_hnsw $efs $2          # SeRF
            else
                echo "search serf at efs=$efs"
                source run_serf_kg.sh query multi_hnsw $efs $2 &          # SeRF
            fi
        done
    elif [ "$run_algo" == "serf_left" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search serf at efs=$efs"
                source run_serf_left.sh query multi_hnsw $efs $2          # SeRF
            else
                echo "search serf at efs=$efs"
                source run_serf_left.sh query multi_hnsw $efs $2 &          # SeRF
            fi
        done
    elif [ "$run_algo" == "serf_right" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search serf at efs=$efs"
                source run_serf_right.sh query multi_hnsw $efs $2          # SeRF
            else
                echo "search serf at efs=$efs"
                source run_serf_right.sh query multi_hnsw $efs $2 &          # SeRF
            fi
        done
    elif [ "$run_algo" == "serf_onethread" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search serf at efs=$efs"
                source run_serf_onethread.sh query multi_hnsw $efs $2          # SeRF
            else
                echo "search serf at efs=$efs"
                source run_serf_onethread.sh query multi_hnsw $efs $2 &          # SeRF
            fi
        done

    elif [ "$run_algo" == "dsg" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search dsg at efs=$efs"
                source run_dsg.sh query multi_hnsw $efs $2          # DSG
            else
                echo "search dsg at efs=$efs"
                source run_dsg.sh query multi_hnsw $efs $2 &          # DSG
            fi
        done

    elif [ "$run_algo" == "dsg_left" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search dsg at efs=$efs"
                source run_dsg_left.sh query multi_hnsw $efs $2          # DSG
            else
                echo "search dsg at efs=$efs"
                source run_dsg_left.sh query multi_hnsw $efs $2 &          # DSG
            fi
        done

    elif [ "$run_algo" == "dsg_right" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search dsg at efs=$efs"
                source run_dsg_right.sh query multi_hnsw $efs $2          # DSG
            else
                echo "search dsg at efs=$efs"
                source run_dsg_right.sh query multi_hnsw $efs $2 &          # DSG
            fi
        done

    elif [ "$run_algo" == "dsg_onethread" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search dsg at efs=$efs"
                source run_dsg_onethread.sh query multi_hnsw $efs $2          # DSG
            else
                echo "search dsg at efs=$efs"
                source run_dsg_onethread.sh query multi_hnsw $efs $2 &          # DSG
            fi
        done
    elif [ "$run_algo" == "vamana_tree" ]; then
        for beamsize in "${beamsize_list[@]}"; do
            last_value=${beamsize_list[-1]}
            if [ $beamsize == $last_value ]; then
                echo "search vamana tree at beamsize=$beamsize"
                source run_vamanatree.sh query multi_vtree $beamsize $2          # WST vamana tree
            else
                echo "search vamana tree at beamsize=$beamsize"
                source run_vamanatree.sh query multi_vtree $beamsize $2 &          # WST vamana tree
            fi
        done

    elif [ "$run_algo" == "wst_sup_opt" ]; then
        for beamsize in "${beamsize_list[@]}"; do
            for final_beam_multiply in "${final_beam_multiply_list[@]}"; do
                last_value=${beamsize_list[-1]}
                last_value2=${final_beam_multiply_list[-1]}
                if [ $beamsize == $last_value ] && [ $final_beam_multiply == $last_value2 ]; then
                    echo "search super opt wst at beamsize=$beamsize, final_beam_multiply=$final_beam_multiply"
                    source run_wst.sh query multi_wst $beamsize $final_beam_multiply $2          # WST super optimized post filtering
                else
                    echo "search super opt wst at beamsize=$beamsize, final_beam_multiply=$final_beam_multiply"
                    source run_wst.sh query multi_wst $beamsize $final_beam_multiply $2 &          # WST super optimized post filtering
                fi
            done
        done

    elif [ "$run_algo" == "unify" ]; then
        for al in "${al_list[@]}"; do
            for ef_search in "${ef_search_list[@]}"; do
                last_value=${ef_search_list[-1]}
                last_value2=${al_list[-1]}
                if [ $ef_search == $last_value ] && [ $al == $last_value2 ]; then
                    echo "search unify at al=$al, ef_search=$ef_search"
                    source run_unify.sh query multi_unify $al $ef_search $2          # UNIFY
                else
                    echo "search unify at al=$al, ef_search=$ef_search"
                    source run_unify.sh query multi_unify $al $ef_search $2 &          # UNIFY
                fi
            done
        done

    elif [ "$run_algo" == "unify_hybrid" ]; then
        for al in "${al_list[@]}"; do
            for ef_search in "${ef_search_list[@]}"; do
                last_value=${ef_search_list[-1]}
                last_value2=${al_list[-1]}
                if [ $ef_search == $last_value ] && [ $al == $last_value2 ]; then
                    echo "search unify hybrid at al=$al, ef_search=$ef_search"
                    source run_unify_hybrid.sh query multi_unify $al $ef_search $2          # UNIFY
                else
                    echo "search unify at al=$al, ef_search=$ef_search"
                    source run_unify_hybrid.sh query multi_unify $al $ef_search $2 &          # UNIFY
                fi
            done
        done
    elif [ "$run_algo" == "unify_bottom" ]; then
        for al in "${al_list[@]}"; do
            for ef_search in "${ef_search_list[@]}"; do
                last_value=${ef_search_list[-1]}
                last_value2=${al_list[-1]}
                if [ $ef_search == $last_value ] && [ $al == $last_value2 ]; then
                    echo "search unify hybrid at al=$al, ef_search=$ef_search"
                    source run_unify_bottom.sh query multi_unify $al $ef_search $2          # UNIFY
                else
                    echo "search unify at al=$al, ef_search=$ef_search"
                    source run_unify_bottom.sh query multi_unify $al $ef_search $2 &          # UNIFY
                fi
            done
        done
    elif [ "$run_algo" == "unify_left" ]; then
        for al in "${al_list[@]}"; do
            for ef_search in "${ef_search_list[@]}"; do
                last_value=${ef_search_list[-1]}
                last_value2=${al_list[-1]}
                if [ $ef_search == $last_value ] && [ $al == $last_value2 ]; then
                    echo "search unify hybrid at al=$al, ef_search=$ef_search"
                    source run_unify_left.sh query multi_unify $al $ef_search $2          # UNIFY
                else
                    echo "search unify at al=$al, ef_search=$ef_search"
                    source run_unify_left.sh query multi_unify $al $ef_search $2 &          # UNIFY
                fi
            done
        done
    elif [ "$run_algo" == "unify_right" ]; then
        for al in "${al_list[@]}"; do
            for ef_search in "${ef_search_list[@]}"; do
                last_value=${ef_search_list[-1]}
                last_value2=${al_list[-1]}
                if [ $ef_search == $last_value ] && [ $al == $last_value2 ]; then
                    echo "search unify hybrid at al=$al, ef_search=$ef_search"
                    source run_unify_right.sh query multi_unify $al $ef_search $2          # UNIFY
                else
                    echo "search unify at al=$al, ef_search=$ef_search"
                    source run_unify_right.sh query multi_unify $al $ef_search $2 &          # UNIFY
                fi
            done
        done
    elif [ "$run_algo" == "unify_middle" ]; then
        for al in "${al_list[@]}"; do
            for ef_search in "${ef_search_list[@]}"; do
                last_value=${ef_search_list[-1]}
                last_value2=${al_list[-1]}
                if [ $ef_search == $last_value ] && [ $al == $last_value2 ]; then
                    echo "search unify hybrid at al=$al, ef_search=$ef_search"
                    source run_unify_middle.sh query multi_unify $al $ef_search $2          # UNIFY
                else
                    echo "search unify at al=$al, ef_search=$ef_search"
                    source run_unify_middle.sh query multi_unify $al $ef_search $2 &          # UNIFY
                fi
            done
        done
    else 
        echo "Invalid multi option: $run_algo"
        exit 1
    fi
}


if [ $1 == "range" ]; then
    # query_func acorn
    # query_func hnsw
    # query_func irange
    # query_func ivfpq
    # query_func milvus_ivfpq
    # query_func milvus_hnsw
    # query_func serf
    # query_func dsg
    query_func vamana_tree
    query_func wst_sup_opt
    query_func unify
    query_func unify_hybrid
    echo "All range benchmarks done."
    exit 0
elif [ $1 == "key" ]; then
    query_func acorn
    query_func diskann
    query_func diskann_stitched
    query_func hnsw
    query_func ivfpq
    query_func kgraph
    query_func nsw
    echo "All label benchmarks done."
    exit 0
elif [ $1 != "batch" ] && [ $1 != "batchkey" ]; then 
    query_func $1
    exit 0
fi






if [ $1 == "batch" ]; then
    if [ $2 == "range" ]; then
        python utils/modify_var.py label_range 100000
        python utils/modify_var.py label_cnt 1
        python utils/modify_var.py query_label_cnt 6
        python utils/modify_var.py query_label 0
        # ./run_attr_generator.sh
        # ./run_qrange_generator.sh
        # ./run_groundtruth_generator.sh
        for qrange in "${sel_list[@]}"; do
            echo "qrange=$qrange"
            python utils/modify_var.py query_label_cnt $qrange
            # ./run_qrange_generator.sh
            ./run_groundtruth_generator.sh
            query_func acorn
            query_func hnsw
            query_func irange
            query_func ivfpq
            query_func milvus_ivfpq
            query_func milvus_hnsw
            query_func serf
            query_func dsg
            query_func vamana_tree
            query_func wst_sup_opt
            query_func unify
            query_func unify_hybrid
            echo "All range benchmarks done"
        done
    elif [ $2 == "label" ]; then
        python utils/modify_var.py label_range 500
        python utils/modify_var.py label_cnt 1
        python utils/modify_var.py query_label_cnt 1
        python utils/modify_var.py query_label 6
        # ./run_attr_generator.sh
        # ./run_qrange_generator.sh
        # ./run_groundtruth_generator.sh
        # ./all_construct.sh keyword
        for q_label in "${sel_list[@]}"; do
            python utils/modify_var.py query_label $q_label
            # ./run_attr_generator.sh
            # ./run_qrange_generator.sh
            ./run_groundtruth_generator.sh
            # ./run_diskann_stitched.sh construction
            query_func acorn
            query_func diskann
            query_func diskann_stitched
            query_func hnsw
            query_func ivfpq
            query_func kgraph
            query_func nsw
            query_func milvus_hnsw
            query_func milvus_ivfpq
            echo "All label benchmarks done."
        done
    elif [ $2 == "arbitrary" ]; then
        python utils/modify_var.py label_range 500
        python utils/modify_var.py label_cnt 2
        python utils/modify_var.py query_label_cnt 6
        python utils/modify_var.py query_label 6
        ./run_attr_generator.sh
        ./run_qrange_generator.sh
        ./run_groundtruth_generator.sh
        ./run_acorn.sh construction
        ./run_hnsw.sh construction
        ./run_ivfpq.sh construction
        ./run_milvus_hnsw.sh construction
        ./run_milvus_ivfpq.sh construction
        for q_label in "${sel_list[@]}"; do
            python utils/modify_var.py query_label_cnt $q_label
            python utils/modify_var.py query_label $q_label
            ./run_qrange_generator.sh
            ./run_groundtruth_generator.sh
            query_func acorn
            query_func hnsw
            query_func ivfpq
            query_func milvus_hnsw
            query_func milvus_ivfpq
            echo "All label benchmarks done."
        done
    else
        for qrange in "${sel_list[@]}"; do
            echo "qrange=$qrange"
            python utils/modify_var.py query_label_cnt $qrange
            ./run_qrange_generator.sh
            ./run_groundtruth_generator.sh
            query_func $2
        done
        echo "All range benchmarks done"
    fi
    exit 0
fi

if [ $1 == "batchkey" ]; then
    if [ $2 == "keyword" ]; then
        python utils/modify_var.py label_range 500
        python utils/modify_var.py label_cnt 1
        python utils/modify_var.py query_label_cnt 1
        python utils/modify_var.py query_label 6
        # ./run_attr_generator.sh
        # ./run_qrange_generator.sh
        # ./run_groundtruth_generator.sh
        # ./all_construct.sh keyword
        for q_label in "${sel_list[@]}"; do
            python utils/modify_var.py query_label $q_label
            ./run_attr_generator.sh
            ./run_qrange_generator.sh
            ./run_groundtruth_generator.sh
            # ./run_diskann_stitched.sh construction
            # query_func acorn
            # query_func diskann
            query_func diskann_stitched
            # query_func hnsw
            # query_func ivfpq
            # query_func kgraph
            # query_func nsw
            # query_func milvus_hnsw
            # query_func milvus_ivfpq
            echo "All label benchmarks done."
        done
    else
        python utils/modify_var.py label_range 500
        python utils/modify_var.py label_cnt 1
        python utils/modify_var.py query_label_cnt 1
        python utils/modify_var.py query_label 6
        for q_label in "${sel_list[@]}"; do
            echo "q_label=$q_label"
            python utils/modify_var.py query_label $q_label
            # ./run_qrange_generator.sh
            query_func $2
        done
        echo "All range benchmarks done"
    fi
    exit 0
fi