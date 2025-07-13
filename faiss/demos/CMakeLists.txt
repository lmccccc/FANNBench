# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(demo_imi_flat EXCLUDE_FROM_ALL demo_imi_flat.cpp)
target_link_libraries(demo_imi_flat PRIVATE faiss)

add_executable(demo_imi_pq EXCLUDE_FROM_ALL demo_imi_pq.cpp)
target_link_libraries(demo_imi_pq PRIVATE faiss)

add_executable(demo_ivfpq_indexing EXCLUDE_FROM_ALL demo_ivfpq_indexing.cpp)
target_link_libraries(demo_ivfpq_indexing PRIVATE faiss)

add_executable(demo_nndescent EXCLUDE_FROM_ALL demo_nndescent.cpp)
target_link_libraries(demo_nndescent PRIVATE faiss)

add_executable(demo_sift1M EXCLUDE_FROM_ALL demo_sift1M.cpp)
target_link_libraries(demo_sift1M PRIVATE faiss)

add_executable(demo_weighted_kmeans EXCLUDE_FROM_ALL demo_weighted_kmeans.cpp)
target_link_libraries(demo_weighted_kmeans PRIVATE faiss)

add_executable(demo_residual_quantizer EXCLUDE_FROM_ALL demo_residual_quantizer.cpp)
target_link_libraries(demo_residual_quantizer PRIVATE faiss)

add_executable(utils EXCLUDE_FROM_ALL utils.cpp)
target_link_libraries(utils PRIVATE faiss)

add_executable(generate_groundtruth EXCLUDE_FROM_ALL generate_groundtruth.cpp)
target_link_libraries(generate_groundtruth PRIVATE faiss)

add_executable(generate_groundtruth_arbi EXCLUDE_FROM_ALL generate_groundtruth_arbi.cpp)
target_link_libraries(generate_groundtruth_arbi PRIVATE faiss)
find_package(OpenMP REQUIRED)
target_link_libraries(generate_groundtruth_arbi PRIVATE OpenMP::OpenMP_CXX)

add_executable(ivfpq_build EXCLUDE_FROM_ALL ivfpq_build.cpp)
target_link_libraries(ivfpq_build PRIVATE faiss)

add_executable(ivfpq_query EXCLUDE_FROM_ALL ivfpq_query.cpp)
target_link_libraries(ivfpq_query PRIVATE faiss)

add_executable(ivfpq_query_keyword EXCLUDE_FROM_ALL ivfpq_query_keyword.cpp)
target_link_libraries(ivfpq_query_keyword PRIVATE faiss)

add_executable(hnsw_build EXCLUDE_FROM_ALL hnsw_build.cpp)
target_link_libraries(hnsw_build PRIVATE faiss)

add_executable(hnsw_query EXCLUDE_FROM_ALL hnsw_query.cpp)
target_link_libraries(hnsw_query PRIVATE faiss)

add_executable(hnsw_query_keyword EXCLUDE_FROM_ALL hnsw_query_keyword.cpp)
target_link_libraries(hnsw_query_keyword PRIVATE faiss)

add_executable(generate_groundtruth_keyword EXCLUDE_FROM_ALL generate_groundtruth_keyword.cpp)
target_link_libraries(generate_groundtruth_keyword PRIVATE faiss)

add_executable(airship_build EXCLUDE_FROM_ALL airship_build.cpp)
target_link_libraries(airship_build PRIVATE faiss)

add_executable(airship_query EXCLUDE_FROM_ALL airship_query.cpp)
target_link_libraries(airship_query PRIVATE faiss)

add_executable(airship_query_keyword EXCLUDE_FROM_ALL airship_query_keyword.cpp)
target_link_libraries(airship_query_keyword PRIVATE faiss)

add_executable(hnsw_query_arbi EXCLUDE_FROM_ALL hnsw_query_arbi.cpp)
target_link_libraries(hnsw_query_arbi PRIVATE faiss)

add_executable(ivfpq_query_arbi EXCLUDE_FROM_ALL ivfpq_query_arbi.cpp)
target_link_libraries(ivfpq_query_arbi PRIVATE faiss)
