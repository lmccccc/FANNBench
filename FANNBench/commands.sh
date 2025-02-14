# ./run_serf.sh construction
# ./run_dsg.sh construction
# ./all_query.sh batch milvus_ivfpq
# ./all_query.sh batch milvus_hnsw
# ./all_query.sh smallbatch serf
# ./all_query.sh smallbatch dsg
# ./all_query.sh batch hnsw
# ./all_query.sh smallbatch hnsw
./run_unify.sh construction
./all_query.sh largebatch unify