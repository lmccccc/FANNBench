run_algo=$1

# search var list
ef_search_list=(
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
)

L_list=(
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
)

nprobe_list=(
    1
    3
    5
    10
    20
    30
    50
    80
    100
    150
    200
    300
)

L_search_list=(
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
)

beamsize_list=(
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
)

final_beam_multiply_list=(
    2
)

al_list=(
    # 8
    16
    32
)


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
                source run_acorn.sh query multi_hnsw $efs          # Acorn
            else
                echo "search acorn at efs=$efs"
                source run_acorn.sh query multi_hnsw $efs &          # Acorn
            fi
        done

    elif [ "$run_algo" == "diskann" ]; then
        for L in "${L_list[@]}"; do
            last_value=${L_list[-1]}
            if [ $L == $last_value ]; then
                echo "search diskann at L=$L"
                source run_diskann.sh query multi_diskann $L          # DiskANN
            else
                echo "search diskann at L=$L"
                source run_diskann.sh query multi_diskann $L &          # DiskANN
            fi    
        done

    elif [ "$run_algo" == "diskann_stitched" ]; then
        for L in "${L_list[@]}"; do
            last_value=${L_list[-1]}
            if [ $L == $last_value ]; then
                echo "search diskann at L=$L"
                source run_diskann_stitched.sh query multi_diskann $L          # DiskANN
            else
                echo "search diskann at L=$L"
                source run_diskann_stitched.sh query multi_diskann $L &          # DiskANN
            fi      
        done

    elif [ "$run_algo" == "hnsw" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search hnsw at efs=$efs"
                source run_hnsw.sh query multi_hnsw $efs          # Faiss HNSW
            else
                echo "search hnsw at efs=$efs"
                source run_hnsw.sh query multi_hnsw $efs &          # Faiss HNSW
            fi
        done

    elif [ "$run_algo" == "irange" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search irange at efs=$efs"
                source run_irange.sh query multi_hnsw $efs          # iRangeGraph
            else
                echo "search irange at efs=$efs"
                source run_irange.sh query multi_hnsw $efs &          # iRangeGraph
            fi
        done

    elif [ "$run_algo" == "ivfpq" ]; then
        for nprobe in "${nprobe_list[@]}"; do
            last_value=${nprobe_list[-1]}
            if [ $nprobe == $last_value ]; then
                echo "search ivfpq at nprobe=$nprobe"
                source run_ivfpq.sh query multi_ivfpq $nprobe          # Faiss IVFPQ
            else
                echo "search ivfpq at nprobe=$nprobe"
                source run_ivfpq.sh query multi_ivfpq $nprobe &          # Faiss IVFPQ
            fi
        done

    elif [ "$run_algo" == "milvus_ivfpq" ]; then 
        for nprobe in "${nprobe_list[@]}"; do
            echo "search milvus ivfpq at nprobe=$nprobe"
            source run_milvus_ivfpq.sh query multi_ivfpq $nprobe # since milvus connects to server, parallel run may cause error
        done

    elif [ "$run_algo" == "milvus_hnsw" ]; then 
        for efs in "${ef_search_list[@]}"; do
            echo "search milvus hnsw at efs=$efs"
            source run_milvus_hnsw.sh query multi_hnsw $efs # since milvus connects to server, parallel run may cause error
        done

    elif [ "$run_algo" == "kgraph" ]; then
        for l_search in "${L_search_list[@]}"; do
            last_value=${L_search_list[-1]}
            if [ $l_search == $last_value ]; then
                echo "search nhq kgraph at L_search=$l_search"
                source run_nhq_kgraph.sh query multi_kgraph $l_search          # NHQ-NPG KGraph
            else
                echo "search nhq kgraph at L_search=$l_search"
                source run_nhq_kgraph.sh query multi_kgraph $l_search &          # NHQ-NPG KGraph
            fi
        done

    elif [ "$run_algo" == "nsw" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search nhq nsw at efs=$efs"
                source run_nhq_nsw.sh query multi_hnsw $efs          # NHQ-NPG NSW
            else
                echo "search nhq nsw at efs=$efs"
                source run_nhq_nsw.sh query multi_hnsw $efs &          # NHQ-NPG NSW
            fi
        done

    elif [ "$run_algo" == "rii" ]; then
        source run_rii.sh query

    elif [ "$run_algo" == "serf" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search serf at efs=$efs"
                source run_serf.sh query multi_hnsw $efs          # SeRF
            else
                echo "search serf at efs=$efs"
                source run_serf.sh query multi_hnsw $efs &          # SeRF
            fi
        done

    elif [ "$run_algo" == "dsg" ]; then
        for efs in "${ef_search_list[@]}"; do
            last_value=${ef_search_list[-1]}
            if [ $efs == $last_value ]; then
                echo "search dsg at efs=$efs"
                source run_dsg.sh query multi_hnsw $efs          # DSG
            else
                echo "search dsg at efs=$efs"
                source run_dsg.sh query multi_hnsw $efs &          # DSG
            fi
        done

    elif [ "$run_algo" == "vamana_tree" ]; then
        for beamsize in "${beamsize_list[@]}"; do
            last_value=${beamsize_list[-1]}
            if [ $beamsize == $last_value ]; then
                echo "search vamana tree at beamsize=$beamsize"
                source run_vamanatree.sh query multi_vtree $beamsize          # WST vamana tree
            else
                echo "search vamana tree at beamsize=$beamsize"
                source run_vamanatree.sh query multi_vtree $beamsize &          # WST vamana tree
            fi
        done

    elif [ "$run_algo" == "wst_sup_opt" ]; then
        for beamsize in "${beamsize_list[@]}"; do
            for final_beam_multiply in "${final_beam_multiply_list[@]}"; do
                last_value=${beamsize_list[-1]}
                last_value2=${final_beam_multiply_list[-1]}
                if [ $beamsize == $last_value ] && [ $final_beam_multiply == $last_value2 ]; then
                    echo "search super opt wst at beamsize=$beamsize, final_beam_multiply=$final_beam_multiply"
                    source run_wst.sh query multi_wst $beamsize $final_beam_multiply          # WST super optimized post filtering
                else
                    echo "search super opt wst at beamsize=$beamsize, final_beam_multiply=$final_beam_multiply"
                    source run_wst.sh query multi_wst $beamsize $final_beam_multiply &          # WST super optimized post filtering
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
                    source run_unify.sh query multi_unify $al $ef_search          # UNIFY
                else
                    echo "search unify at al=$al, ef_search=$ef_search"
                    source run_unify.sh query multi_unify $al $ef_search &          # UNIFY
                fi
            done
        done

    else 
        echo "Invalid multi option."
        exit 1
    fi
}


if [ $1 == "all" ]; then
    query_func acorn
    query_func diskann
    query_func diskann_stitched
    query_func hnsw
    query_func irange
    query_func ivfpq
    query_func milvus_ivfpq
    query_func milvus_hnsw
    query_func kgraph
    query_func nsw
    query_func serf
    query_func dsg
    query_func vamana_tree
    query_func wst_sup_opt
    query_func unify
    echo "All benchmarks done."
elif [ $1 == "allrange" ]; then
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
    echo "All range benchmarks done."
elif [ $1 == "allkey" ]; then
    query_func acorn
    query_func diskann
    query_func diskann_stitched
    query_func hnsw
    query_func ivfpq
    query_func kgraph
    query_func nsw
    echo "All label benchmarks done."
else
    query_func $1
fi



# echo "Run all benchmarks"
# run_func run_acorn.sh $1          # Acorn
# run_func run_diskann.sh $1        # DiskANN
# run_func run_hnsw.sh $1           # Faiss HNSW
# run_func run_irange.sh $1         # iRangeGraph
# run_func run_ivfpq.sh $1          # Faiss IVFPQ
# run_func run_milvus_ivfpq.sh $1   # Milvus IVFPQ
# run_func run_nhq_kgraph.sh $1     # NHQ-NPG KGraph
# run_func run_nhq_nsw.sh $1        # NHQ-NPG NSW
# run_func run_rii.sh $1            # RII
# run_func run_serf.sh $1           # SeRF
# run_func run_vamanatree.sh $1     # WST vamana tree
# run_func run_wst.sh $1            # WST super optimized post filtering

