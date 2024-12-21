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

if [ "$run_algo" == "acorn" ]; then
    for efs in "${ef_search_list[@]}"; do
        echo "search acorn at efs=$efs"
        source run_acorn.sh query multi_hnsw $efs &          # Acorn
    done

elif [ "$run_algo" == "diskann" ]; then
    for L in "${L_list[@]}"; do
        echo "search diskann at L=$L"
        source run_diskann.sh query multi_diskann $L &          # Acorn
    done

elif [ "$run_algo" == "hnsw" ]; then
    for efs in "${ef_search_list[@]}"; do
        echo "search hnsw at efs=$efs"
        source run_hnsw.sh query multi_hnsw $efs &
    done

elif [ "$run_algo" == "irange" ]; then
    for efs in "${ef_search_list[@]}"; do
        echo "search irange at efs=$efs"
        source run_irange.sh query multi_hnsw $efs &
    done

elif [ "$run_algo" == "ivfpq" ]; then
    for nprobe in "${nprobe_list[@]}"; do
        echo "search ivfpq at nprobe=$nprobe"
        source run_ivfpq.sh query multi_ivfpq $nprobe &
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
        echo "search nhq kgraph at L_search=$l_search"
        source run_nhq_kgraph.sh query multi_kgraph $l_search &
    done

elif [ "$run_algo" == "nsw" ]; then
    for efs in "${ef_search_list[@]}"; do
        echo "search irange at efs=$efs"
        source run_nhq_nsw.sh query multi_hnsw $efs &
    done

elif [ "$run_algo" == "rii" ]; then
    source run_rii.sh query

elif [ "$run_algo" == "serf" ]; then
    for efs in "${ef_search_list[@]}"; do
        echo "search irange at efs=$efs"
        source run_serf.sh query multi_hnsw $efs &
    done

elif [ "$run_algo" == "vamana_tree" ]; then
    for beamsize in "${beamsize_list[@]}"; do
        echo "search irange at beamsize=$beamsize"
        source run_vamanatree.sh query multi_vtree $beamsize &
    done

elif [ "$run_algo" == "wst_sup_opt" ]; then
    for beamsize in "${beamsize_list[@]}"; do
        for final_beam_multiply in "${final_beam_multiply_list[@]}"; do
            echo "search super opt wst at beamsize=$beamsize, final_beam_multiply=$final_beam_multiply"
            source run_wst.sh query multi_wst $beamsize $final_beam_multiply &
        done
    done

elif [ "$run_algo" == "test" ]; then
    for l_search in "${L_search_list[@]}"; do
        source test_multi.sh query multi_kgraph $l_search &
    done


else 
    echo "Invalid multi option."
    exit 1
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

