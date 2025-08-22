# ./run_serf.sh construction
# ./run_dsg.sh construction
# ./all_query.sh batch milvus_ivfpq
# ./all_query.sh batch milvus_hnsw
# ./all_query.sh smallbatch serf
# ./all_query.sh smallbatch dsg
# ./all_query.sh batch hnsw
# ./all_query.sh smallbatch hnsw
# ./run_unify.sh construction

# ./all_query.sh batch range
# ./run_milvus_hnsw.sh construction
# ./run_ivfpq.sh construction
# ./run_milvus_ivfpq.sh construction
# python utils/modify_var.py K 100
# python utils/modify_var.py dataset spacev
# ./all_query.sh batch range
# ./run_ivfpq.sh construction
# ./run_milvus_ivfpq.sh construction
# python utils/modify_var.py dataset redcaps
# ./run_ivfpq.sh construction
# ./run_milvus_ivfpq.sh construction
# python utils/modify_var.py dataset youtube
# ./run_ivfpq.sh construction
# ./run_milvus_ivfpq.sh construction


./run_milvus_hnsw.sh construction
./run_milvus_ivfpq.sh construction
./all_query.sh batch label

