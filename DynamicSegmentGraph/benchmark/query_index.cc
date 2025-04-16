/**
 * @file exp_halfbound.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Benchmark Half-Bounded Range Filter Search
 * @date 2023-12-22
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>
#include <tuple>

#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "segment_graph_2d.h"
#include "compact_graph.h"
#include "utils.h"
#include <iomanip>

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

void log_result_recorder(
    const std::map<int, std::tuple<double, double, double, double>> &result_recorder,
    const std::map<int, std::tuple<float, float>> &comparison_recorder,
    const int amount) {
    double total_recall = 0, total_comps = 0, total_qps = 0, total_hops = 0, total_fetch = 0;
    for (auto item : result_recorder) {
        const auto &[recall, calDistTime, internal_search_time, fetch_nn_time] = item.second;
        const auto &[comps, hops] = comparison_recorder.at(item.first);
        const auto cur_range_amount = amount / result_recorder.size();
        total_recall += recall;
        total_comps += comps;
        total_qps += internal_search_time;
        total_hops += hops;
        total_fetch += fetch_nn_time;
        cout << std::setiosflags(ios::fixed) << std::setprecision(4)
             << "range: " << item.first
             << "\t recall: " << recall / cur_range_amount
             << "\t QPS: " << std::setprecision(0)
             << cur_range_amount / internal_search_time << "\t"
             << "Comps: " << comps / cur_range_amount << std::setprecision(4)
             << "\t Hops: " << hops / cur_range_amount << std::setprecision(4) << std::endl;
        //  << "\t Internal Search Time: " << internal_search_time
        //  << "\t Fetch NN Time: " << fetch_nn_time
        //  << "\t CalDist Time: " << calDistTime << std::endl; // 新增一行显示CalDist时间
    }
    cout << "Total Recall: " << total_recall / amount
         << "\t Total QPS: " << amount / total_qps
         << "\t Total Comps: " << total_comps / amount
         << "\t Total Hops: " << total_hops / amount << std::endl;
    cout << "Fetch percentage: " << total_fetch / total_qps << std::endl;
}

vector<int> get_label(int N, string file_name){  //read json file, only [1,\n2,\n3,\n...]
  vector<int> label;
  std::ifstream in(file_name);
  if (!in.is_open()) {
    std::cerr << "Error: failed to open file " << file_name << std::endl;
    exit(-1);
  }
  int temp;
  char c;
  in >> c;
  assert(c == '[');
  int i = 0;
  while (true) {
    //get head of ifstream  
    if(in.peek()==','){
      in >> c;
      continue;
    }
    in >> temp;
    label.push_back(temp);
    i++;
    if(i == N){
      break;
    }
  }

  in.close();
  return label;
}


vector<pair<int, int>> get_range(int N, string file_name){  //read text file, one column, one integer each row
  vector<int> label;
  std::ifstream in(file_name);
  if (!in.is_open()) {
    std::cerr << "Error: failed to open file " << file_name << std::endl;
    exit(-1);
  }
  int temp;
  char c;
  in >> c;
  assert(c == '[');
  int i = 0;
  while (true) {
    //get head of ifstream  
    if(in.peek()==','){
      in >> c;
      continue;
    }
    in >> temp;
    label.push_back(temp);
    i++;
    if(i == N * 2){
      break;
    }
  }
  in.close();

  vector<pair<int, int>> qrange;
  for(int i = 0; i < N * 2; i+=2){
    qrange.push_back(make_pair(label[i], label[i+1]));
  }
  return qrange;
}

vector<int> sort_data_by_label(vector<vector<float>> &data, vector<vector<int>>& groundtruth, vector<int> &label, int N, int k){//txt file, 1\n2\n3\n...
  vector<int> od2id(N);
  vector<int> id2od(N);
  std::iota(od2id.begin(), od2id.end(), 0);//init ori id
  std::sort(od2id.begin(), od2id.end(), [&](int i, int j) { return label[i] < label[j]; });
  vector<vector<float>> data_sorted(N);
  vector<int> label_sorted(N);
  vector<vector<int>> groundtruth_sorted(groundtruth.size());
  for (int i = 0; i < N; i++) {
    data_sorted[i] = data[od2id[i]];
    label_sorted[i] = label[od2id[i]];

    id2od[od2id[i]] = i;
  }
  for (int i = 0; i < groundtruth.size(); ++i){
    for (int j = 0; j < k; ++j){
      groundtruth[i][j] = id2od[groundtruth[i][j]];
    }
  }
  //copy data_soorted to data 
  data.assign(data_sorted.begin(), data_sorted.end());
  label.assign(label_sorted.begin(), label_sorted.end());
  return od2id;
}

vector<vector<int>> load_groundtruth(string file_name, int Nq, int K){//json file
  vector<vector<int>> gt;
  std::ifstream in(file_name);
  if (!in.is_open()) {
    std::cerr << "Error: failed to open file " << file_name << std::endl;
    exit(-1);
  }
  int temp;
  char c;
  in >> c;
  assert(c == '[');
  int i = 0;
  vector<int> tmp_gt;
  while (true) {
    //get head of ifstream  
    if(in.peek()==','){
      in >> c;
      continue;
    }
    in >> temp;
    tmp_gt.push_back(temp);
    i++;
    if(i % K == 0){
      gt.push_back(tmp_gt);
      tmp_gt.clear();
    }
    if(i == Nq * K){
      break;
    }
  }
  in.close();
  return gt;
}

// 找到 vector 中大于等于 val 的第一个元素的位置
int findFirstGreaterOrEqualPos(const std::vector<int>& vec, int val) {
  auto it = std::lower_bound(vec.begin(), vec.end(), val);
  if (it == vec.end()) {
      return -1; // 表示未找到
  }
  return std::distance(vec.begin(), it);
}

// 找到 vector 中小于等于 val 的最后一个元素的位置
int findLastLessOrEqualPos(const std::vector<int>& vec, int val) {
  auto it = std::upper_bound(vec.begin(), vec.end(), val);
  if (it == vec.begin()) {
      return -1; // 表示未找到
  }
  --it;
  return std::distance(vec.begin(), it);
}

//convert label range to id range, only support ordered integer label range from 0 to N-1
void label_range_2_id_range(vector<pair<int, int>> &qrange, vector<int> &sorted_label){
  for (size_t i = 0; i < qrange.size(); i++)
  {
    qrange[i].first = findFirstGreaterOrEqualPos(sorted_label, qrange[i].first);
    qrange[i].second = findLastLessOrEqualPos(sorted_label, qrange[i].second);
  }
  
  // vector<int> label_start, label_end;
  // int previous_label = -1;
  // for (size_t i = 0; i < sorted_label.size(); i++)
  // {
  //   if(label_start.size() == 0) {
  //     label_start.push_back(i);
  //   }
  //   else if(sorted_label[i] != sorted_label[i-1]){
  //     label_end.push_back(i-1);
  //     label_start.push_back(i);
  //   }
  //   if(i == sorted_label.size() - 1){
  //     label_end.push_back(i);
  //   }
  // }
  // cout << "label range size:" << label_start.size() << endl;
  // assert(label_start.size() == label_end.size());
  // for (size_t i = 0; i < qrange.size(); i++)
  // {
  //   qrange[i].first = label_start[qrange[i].first];
  //   qrange[i].second = label_end[qrange[i].second];
  // }
}

int main(int argc, char **argv) {
#ifdef USE_SSE
    cout << "Use SSE" << endl;
#endif

    // Parameters
    string dataset = "deep";
    int data_size = 100000;
    string dataset_path = "";
    string method = "";
    string query_path = "";
    string qrange_path = "";
    string groundtruth_path = "";
    string label_path = "";
    int query_num = 1000;
    int query_k = 10;
    unsigned index_k = 8;
    unsigned ef_max = 500;
    unsigned ef_construction = 100;
    int nthreads = 1;
    int efs = 100;

    string index_path;
    string version = "Benchmark";
    vector<int> od2id;//store original id for recall computation

    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") dataset = string(argv[i + 1]);
        if (arg == "-N")
            data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path")
            dataset_path = string(argv[i + 1]);
        if (arg == "-query_path")
            query_path = string(argv[i + 1]);
        if (arg == "-groundtruth_path")
            groundtruth_path = string(argv[i + 1]);
        if (arg == "-method")
            method = string(argv[i + 1]);
        if (arg == "-index_path")
            index_path = string(argv[i + 1]);
        if (arg == "-k")
            index_k = atoi(argv[i + 1]);
        if (arg == "-ef_max")
            ef_max = atoi(argv[i + 1]);
        if (arg == "-ef_construction")
            ef_construction = atoi(argv[i + 1]);
        if (arg == "-nthreads")
            nthreads = atoi(argv[i + 1]);
        if (arg == "-ef_search")
            efs = atoi(argv[i + 1]);
        if (arg == "-qrange_path")
            qrange_path = string(argv[i + 1]);
        if (arg == "-query_num")
            query_num = atoi(argv[i + 1]);
        if (arg == "-label_path")
            label_path = string(argv[i + 1]);
        if (arg == "-query_k")
            query_k = atoi(argv[i + 1]);
    }

    std::cout << "dataset: " << dataset << endl;
    std::cout << "data_size: " << data_size << endl;
    std::cout << "dataset_path: " << dataset_path << endl;
    std::cout << "query_path: " << query_path << endl;
    std::cout << "groundtruth_path: " << groundtruth_path << endl;
    std::cout << "method: " << method << endl;
    std::cout << "index_k: " << index_k << endl;
    std::cout << "ef_max: " << ef_max << endl;
    std::cout << "ef_construction: " << ef_construction << endl;
    std::cout << "nthreads: " << nthreads << endl;
    std::cout << "ef_search: " << efs << endl;
    std::cout << "qrange_path: " << qrange_path << endl;
    std::cout << "query_num: " << query_num << endl;
    std::cout << "label_path: " << label_path << endl;
    std::cout << "query_k: " << query_k << endl;


    DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);
    cout << "get data size:" << data_wrapper.nodes.size() << " query size:" << data_wrapper.querys.size() << endl;
    data_wrapper.nodes_keys = get_label(data_size, label_path);
    cout << "get data label size:" << data_wrapper.nodes_keys.size() << endl;
    data_wrapper.query_ranges = get_range(query_num, qrange_path);
    cout << "get query range size:" << data_wrapper.query_ranges.size() << endl;

    //Load groundtruth by new method
    data_wrapper.groundtruth = load_groundtruth(groundtruth_path, data_wrapper.query_ranges.size(), query_k);
    cout << "get groundtruth size:" << data_wrapper.groundtruth.size() << endl;
    
    data_wrapper.query_ids.resize(data_wrapper.querys.size());
    std::iota(data_wrapper.query_ids.begin(), data_wrapper.query_ids.end(), 0);//init ori id
    cout << "get query id size:" << data_wrapper.query_ids.size() << endl;
    cout << "query range size:" << data_wrapper.query_ranges.size() << endl;
    assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());
    // data_wrapper.LoadGroundtruth(groundtruth_path);
    // assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());

    timeval t1, t2;
    gettimeofday(&t1, NULL);
    od2id = sort_data_by_label(data_wrapper.nodes, data_wrapper.groundtruth, data_wrapper.nodes_keys, data_size, query_k);
    label_range_2_id_range(data_wrapper.query_ranges, data_wrapper.nodes_keys);
    gettimeofday(&t2, NULL);
    logTime(t1, t2, "Sort Dataset Time");

    // int st = 16;     // starting value
    // int ed = 400;    // ending value (inclusive)
    // int stride = 16; // stride value
    std::vector<int> searchef_para_range_list;
    searchef_para_range_list.push_back(efs);
    // for (int i = st; i <= ed; i += stride) {
    //     searchef_para_range_list.push_back(i);
    // }

    cout << "search ef:" << endl;
    print_set(searchef_para_range_list);

    data_wrapper.version = version;

    base_hnsw::L2Space ss(data_wrapper.data_dim);

    // timeval t1, t2;

    BaseIndex::IndexParams i_params(index_k, ef_construction,
                                    ef_construction, ef_max);
    BaseIndex* index;
    if(method == "Seg2D"){
        index = new SeRF::IndexSegmentGraph2D(&ss, &data_wrapper);
    }else{
        index = new Compact::IndexCompactGraph(&ss, &data_wrapper);
    }
    BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D",
                                      "benchmark");
    timeval t3, t4;
    gettimeofday(&t3, NULL);
    index->load(index_path);
    gettimeofday(&t4, NULL);
    logTime(t3, t4, "Load Index Time");
    cout << "load index done" << endl;
    {
        timeval tt3, tt4;
        BaseIndex::SearchParams s_params;
        s_params.query_K = data_wrapper.query_k;
        for (auto one_searchef : searchef_para_range_list) {
            s_params.search_ef = one_searchef;
            std::map<int, std::tuple<double, double, double, double>> result_recorder; // first->precision, second-> caldist time, third->query_time
            std::map<int, std::tuple<float, float>> comparison_recorder;
            gettimeofday(&tt3, NULL);
            for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
                int one_id = data_wrapper.query_ids.at(idx);
                s_params.query_range =
                    data_wrapper.query_ranges.at(idx).second - data_wrapper.query_ranges.at(idx).first + 1;
                if(method == "Seg2D"){
                    auto res = index->rangeFilteringSearchOutBound(
                    &s_params, &search_info, data_wrapper.querys.at(one_id),
                    data_wrapper.query_ranges.at(idx));
                search_info.precision =
                    countPrecision(data_wrapper.groundtruth.at(idx), res);
                }else{
                    auto res = index->rangeFilteringSearchInRange(
                    &s_params, &search_info, data_wrapper.querys.at(one_id),
                    data_wrapper.query_ranges.at(idx));
                search_info.precision =
                    countPrecision(data_wrapper.groundtruth.at(idx), res);
                }
                
                std::get<0>(result_recorder[s_params.query_range]) += search_info.precision;
                std::get<1>(result_recorder[s_params.query_range]) += search_info.cal_dist_time;
                std::get<2>(result_recorder[s_params.query_range]) += search_info.internal_search_time;
                std::get<3>(result_recorder[s_params.query_range]) += search_info.fetch_nns_time;
                std::get<0>(comparison_recorder[s_params.query_range]) += search_info.total_comparison;
                std::get<1>(comparison_recorder[s_params.query_range]) += search_info.path_counter;
            }

            cout << endl
                 << "Search ef: " << one_searchef << endl
                 << "========================" << endl;
            log_result_recorder(result_recorder, comparison_recorder,
                                data_wrapper.query_ids.size());
            cout << "========================" << endl;
            logTime(tt3, tt4, "total query time");
        }
    }

    return 0;
}