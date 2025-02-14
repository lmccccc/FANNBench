run_algo=$1

# search var list
ef_search_list=(
    10
    12
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
    # 1
    # 3
    # 5
    # 10
    20
    30
    50
    80
    100
    150
    20
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
    10
    12
    15
    18
    20
    40
    60
    80
    100
    150
    200
    # 300
    # 400
    # 500
    # 1000
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
                source run_acorn.sh query multi_hnsw $efs $2          # Acorn
            else
                echo "search acorn at efs=$efs"
                source run_acorn.sh query multi_hnsw $efs $2 &          # Acorn
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
    exit 0
elif [ $1 == "range" ]; then
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
elif [ $1 != "batch" ] && [ $1 != "smallbatch" ] && [ $1 != "largebatch" ]; then 
    query_func $1
    exit 0
fi



qrange_list=(
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    10000
)


if [ $1 == "batch" ]; then
    for qrange in "${qrange_list[@]}"; do
        echo "qrange=$qrange"
        ./run_qrange_generator.sh batch_qr $qrange
        ./run_groundtruth_generator.sh batch_gt $qrange
        if [ $2 == "all" ]; then
            query_func acorn $qrange
            query_func diskann $qrange
            query_func diskann_stitched $qrange
            query_func hnsw $qrange
            query_func irange $qrange
            query_func ivfpq $qrange
            query_func milvus_ivfpq $qrange
            query_func milvus_hnsw $qrange
            query_func kgraph $qrange
            query_func nsw $qrange
            query_func serf $qrange
            query_func dsg $qrange
            query_func vamana_tree $qrange
            query_func wst_sup_opt $qrange
            query_func unify $qrange
            echo "All benchmarks done."
        elif [ $2 == "range" ]; then
            # query_func acorn $qrange
            # query_func hnsw $qrange
            # query_func irange $qrange
            # query_func ivfpq $qrange
            query_func milvus_ivfpq $qrange
            # query_func milvus_hnsw $qrange
            # query_func serf $qrange
            # query_func dsg $qrange
            # query_func vamana_tree $qrange
            # query_func wst_sup_opt $qrange
            # query_func unify $qrange
            echo "All range benchmarks done"
        elif [ $2 == "key" ]; then
            query_func acorn $qrange
            query_func diskann $qrange
            query_func diskann_stitched $qrange
            query_func hnsw $qrange
            query_func ivfpq $qrange
            query_func kgraph $qrange
            query_func nsw $qrange
            echo "All label benchmarks done."
        else
            query_func $2 $qrange
        fi
    done
    exit 0
fi

smallqrange_list=(
    100
    200
    300
    400
    500
    600
    700
    800
    900
)

if [ $1 == "smallbatch" ]; then
    for qrange in "${smallqrange_list[@]}"; do
        echo "qrange=$qrange"
        ./run_qrange_generator.sh batch_qr $qrange
        ./run_groundtruth_generator.sh batch_gt $qrange
        if [ $2 == "all" ]; then
            query_func acorn $qrange
            query_func diskann $qrange
            query_func diskann_stitched $qrange
            query_func hnsw $qrange
            query_func irange $qrange
            query_func ivfpq $qrange
            query_func milvus_ivfpq $qrange
            query_func milvus_hnsw $qrange
            query_func kgraph $qrange
            query_func nsw $qrange
            query_func serf $qrange
            query_func dsg $qrange
            query_func vamana_tree $qrange
            query_func wst_sup_opt $qrange
            query_func unify $qrange
            echo "All benchmarks done."
        elif [ $2 == "range" ]; then
            query_func acorn $qrange
            query_func hnsw $qrange
            query_func irange $qrange
            query_func ivfpq $qrange
            # query_func milvus_ivfpq $qrange
            # query_func milvus_hnsw $qrange
            query_func serf $qrange
            query_func dsg $qrange
            query_func vamana_tree $qrange
            query_func wst_sup_opt $qrange
            query_func unify $qrange
            echo "All range benchmarks done"
        elif [ $2 == "key" ]; then
            query_func acorn $qrange
            query_func diskann $qrange
            query_func diskann_stitched $qrange
            query_func hnsw $qrange
            query_func ivfpq $qrange
            query_func kgraph $qrange
            query_func nsw $qrange
            echo "All label benchmarks done."
        else
            query_func $2 $qrange
        fi
    done
    exit 0
fi


largeqrange_list=(
    20000
    30000
    40000
    50000
    60000
    70000
    80000
    90000
    100000
)

if [ $1 == "largebatch" ]; then
    for qrange in "${largeqrange_list[@]}"; do
        echo "qrange=$qrange"
        ./run_qrange_generator.sh batch_qr $qrange
        ./run_groundtruth_generator.sh batch_gt $qrange
        if [ $2 == "all" ]; then
            query_func acorn $qrange
            query_func diskann $qrange
            query_func diskann_stitched $qrange
            query_func hnsw $qrange
            query_func irange $qrange
            query_func ivfpq $qrange
            query_func milvus_ivfpq $qrange
            query_func milvus_hnsw $qrange
            query_func kgraph $qrange
            query_func nsw $qrange
            query_func serf $qrange
            query_func dsg $qrange
            query_func vamana_tree $qrange
            query_func wst_sup_opt $qrange
            query_func unify $qrange
            echo "All benchmarks done."
        elif [ $2 == "range" ]; then
            query_func acorn $qrange
            # query_func hnsw $qrange
            query_func irange $qrange
            # query_func ivfpq $qrange
            # query_func milvus_ivfpq $qrange
            # query_func milvus_hnsw $qrange
            query_func serf $qrange
            query_func dsg $qrange
            query_func vamana_tree $qrange
            query_func wst_sup_opt $qrange
            query_func unify $qrange
            echo "All range benchmarks done"
        elif [ $2 == "key" ]; then
            query_func acorn $qrange
            query_func diskann $qrange
            query_func diskann_stitched $qrange
            query_func hnsw $qrange
            query_func ivfpq $qrange
            query_func kgraph $qrange
            query_func nsw $qrange
            echo "All label benchmarks done."
        else
            query_func $2 $qrange
        fi
    done
    exit 0
fi

