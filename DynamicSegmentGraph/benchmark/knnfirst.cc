/**
 * @file exp_two_side_pos.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Sensitivity Experiment of Twoside Segment Graph
 * @date 2023-07-07
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "baselines/knn_first_hnsw.h"
#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "utils.h"

// #define LOG_DEBUG_MODE 1

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

long long before_memory, after_memory;

std::vector<std::vector<unsigned>> generatePermutations(const std::vector<unsigned> &batch_sizes) {
    std::vector<std::vector<unsigned>> permutations;
    unsigned start = 0; // Starting point for the first batch
    if (batch_sizes.size() > 1)
        assert(batch_sizes[0] < batch_sizes[1]);
    // Seed the random number generator with current time
    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (unsigned size : batch_sizes) {
        std::vector<unsigned> batch;

        // Generate numbers from 'start' to 'size - 1'
        for (unsigned i = start; i < size; ++i) {
            batch.push_back(i);
        }

        // Shuffle to create a random permutation
        std::shuffle(batch.begin(), batch.end(), rng);

        // Store the permutation and update the starting point for the next batch
        permutations.push_back(batch);
        start = size;
    }

    return permutations;
}

// Function to replace a substring in all elements of a vector of strings
void ReplaceSubstringInPaths(std::vector<std::string> &paths, const std::string &old_str, const std::string &new_str) {
    for (std::string &path : paths) {
        size_t pos = path.find(old_str);
        if (pos != std::string::npos) {
            path.replace(pos, old_str.length(), new_str);
        }
    }
}

int data_size = 1200000;
vector<unsigned> batches_size = {1000000, 1050000, 1100000, 1150000, 1200000};
auto insert_batches = generatePermutations(batches_size);
vector<string> gt_paths = {
    "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/wiki-image_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs",
    "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/wiki-image_benchmark-groundtruth-deep-1050k-num1000-k10.arbitrary.cvs",
    "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/wiki-image_benchmark-groundtruth-deep-1100k-num1000-k10.arbitrary.cvs",
    "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/wiki-image_benchmark-groundtruth-deep-1150k-num1000-k10.arbitrary.cvs",
    "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/wiki-image_benchmark-groundtruth-deep-1200k-num1000-k10.arbitrary.cvs"};

// int data_size = 10000;
// vector<unsigned> batches_size = {1000, 10000};
// auto insert_batches = generatePermutations(batches_size);
// vector<string> gt_paths = {
//     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/deep_benchmark-groundtruth-deep-1k-num1000-k10.arbitrary.cvs",
//     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/deep_benchmark-groundtruth-deep-10k-num1000-k10.arbitrary.cvs"};

int exp(string dataset, int data_size, string dataset_path, string query_path, const bool is_amarel, string gt_path, unsigned index_k, unsigned ef_construction, string version) {
#ifdef USE_SSE
    cout << "Use SSE" << endl;
#endif

    cout << "Baseline knn-first with incremental searching HNSW" << endl;

    int query_k = 10;
    int query_num = 1000;
    DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);
    data_wrapper.LoadGroundtruth(gt_path);
    vector<int> searchef_para_range_list = {100, 200, 300, 400};

    version = "search-" + version;
    if (is_amarel) {
        version = "amarel-" + version;
    }

    string halfbound_gt_path = "";
    // assert(data_size == 1000000);

    cout << "querys num:" << data_wrapper.query_ids.size() << endl;
    cout << "querys num:" << data_wrapper.nodes.size() << endl;

    assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());
    base_hnsw::L2Space ss(data_wrapper.data_dim);

    timeval t1, t2, t3, t4;

    data_wrapper.version = version;

    BaseIndex::IndexParams i_params;
    i_params.ef_construction = ef_construction;
    i_params.K = index_k;
    // i_params.ef_large_for_pruning = 0;
    i_params.ef_max = 0;
    {
        // Baseline
        cout << endl;

        KnnFirstWrapper index(&data_wrapper);
        auto * ss = new hnswlib_incre::L2Space(data_wrapper.data_dim);
        index.initForBuilding(&i_params, ss);

        for (int i = 0; i < insert_batches.size(); i++) {
            auto &insert_batch = insert_batches[i];
        auto &gt_path = gt_paths[i];

            gettimeofday(&t1, NULL);
            index.insertBatch(insert_batch);
            gettimeofday(&t2, NULL);
            logTime(t1, t2, "Build knnfirst HNSW Index Time");

            data_wrapper.LoadGroundtruth(gt_path);
            cout << "HNSW Total # of Neighbors: " << index.index_info->nodes_amount
             << endl;

            cout << "twoside" << endl;
            BaseIndex::SearchInfo search_info(&data_wrapper, &i_params,
                                              "KnnFirstHnsw",
                                              "knnfirst/twoside");
            execute_knn_first_search(index, search_info, data_wrapper,
                                     searchef_para_range_list);

        }
    }

    return 0;
}

int main(int argc, char **argv) {
    string dataset = "biggraph";
    int data_size = 100000;
    string dataset_path = "";
    string method = "";
    string query_path = "";
    bool is_amarel = false;
    string groundtruth_path = "";
    unsigned index_k = 16;
    unsigned ef_construction = 100;
    string indexk_str = "";
    string ef_con_str = "";
    string ef_max_str = "";
    string version = "";
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") dataset = string(argv[i + 1]);
        if (arg == "-N") data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
        if (arg == "-query_path") query_path = string(argv[i + 1]);
        if (arg == "-amarel") is_amarel = true;
        if (arg == "-groundtruth_path") groundtruth_path = string(argv[i + 1]);
        if (arg == "-k")
            index_k = atoi(argv[i + 1]);
        if (arg == "-ef_construction")
            ef_construction = atoi(argv[i + 1]);
        if (arg == "-version") version = string(argv[i + 1]);
    }
    if (dataset != "wiki-image") {
        ReplaceSubstringInPaths(gt_paths, "wiki-image", dataset);
        cout << "Print the first groundtruth path" << gt_paths[0] << endl;
    }

    cout << "index K:" << index_k<< endl;
    cout << "ef construction:" <<ef_construction<< endl;
    // assert(groundtruth_path != "");

    exp(dataset, data_size, dataset_path, query_path, is_amarel, groundtruth_path,
        index_k, ef_construction, version);
    return 0;
}