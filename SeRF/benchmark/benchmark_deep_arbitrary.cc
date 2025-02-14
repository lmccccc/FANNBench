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

// #include "baselines/knn_first_hnsw.h"
#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "segment_graph_2d.h"
#include "utils.h"

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif
#include <iomanip>

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

void log_result_recorder(
    const std::map<int, std::pair<float, float>> &result_recorder,
    const std::map<int, float> &comparison_recorder, const int amount) {
  for (auto it : result_recorder) {
    cout << std::setiosflags(ios::fixed) << std::setprecision(4)
         << "range: " << it.first
         << "\t recall: " << it.second.first / (amount / result_recorder.size())
         << "\t QPS: " << std::setprecision(0)
         << (amount / result_recorder.size()) / it.second.second << "\t Comps: "
         << comparison_recorder.at(it.first) / (amount / result_recorder.size())
         << endl;
  }
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

// binary search in sorted label
// if suc return index, else return internal position
int binary_search(vector<int> &sorted_label, int target){
  int left = 0, right = sorted_label.size() - 1;
  while(left <= right){
    int mid = left + (right - left) / 2;
    if(sorted_label[mid] == target){
      return mid;
    }
    else if(sorted_label[mid] < target){
      left = mid + 1;
    }
    else{
      right = mid - 1;
    }
  }
  return left;
}

//convert label range to id range, only support ordered integer label range from 0 to N-1
void label_range_2_id_range(vector<pair<int, int>> &qrange, vector<int> &sorted_label){
  for (size_t i = 0; i < qrange.size(); i++)
  {
    qrange[i].first = binary_search(sorted_label, qrange[i].first);
    qrange[i].second = binary_search(sorted_label, qrange[i].second);
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
  int thread = 1;
  string dataset_path = "";
  string method = "";
  string query_path = "";
  string groundtruth_path = "";
  string label_path = "";
  string qrange_path = "";
  string ef_search_str = "";
  string index_file = "";
  // vector<int> index_k_list = {8};
  // vector<int> ef_construction_list = {100};
  int query_num = 1000;
  int query_k = 10;
  int ef_construction = -1;
  int _ef_max = -1;
  int M = -1;
  vector<int> ef_max_list = {500};

  string indexk_str = "";
  string ef_con_str = "";
  // string ef_max_str = "";
  string version = "Benchmark";

  //extra info for data label query and sort
  vector<int> od2id;//store original id for recall computation
  if (argc != 33){
    cout << "Usage: " << argv[0] << " -N data_size -dataset_path dataset_path -query_path query_path -label_path label_path -qrange_path qrange_path -groundtruth_path groundtruth_path -method method -dataset dataset -query_num query_num -K query_k -ef_construction ef_construction -index_file index_file" << endl;
    cout << "error argc: " << argc << endl;
    exit(-1);
  }
  for (int i = 1; i < argc; i+=2) {
    string arg = argv[i];
    // if (arg == "-dataset") dataset = string(argv[i + 1]);
    if (arg == "-N") data_size = atoi(argv[i + 1]);
    else if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
    else if (arg == "-query_path") query_path = string(argv[i + 1]);
    else if (arg == "-label_path") label_path = string(argv[i + 1]);
    else if (arg == "-qrange_path") qrange_path = string(argv[i+1]);
    else if (arg == "-groundtruth_path") groundtruth_path = string(argv[i + 1]);
    // if (arg == "-ef_max") ef_max_str = string(argv[i + 1]);
    else if (arg == "-method") method = string(argv[i + 1]);
    else if (arg == "-dataset") dataset = string(argv[i+1]);
    else if (arg == "-query_num") query_num = atoi(argv[i + 1]);
    else if (arg == "-K") query_k = atoi(argv[i + 1]);
    else if  (arg == "-ef_construction") ef_construction = atoi(argv[i + 1]);
    else if  (arg == "-ef_max") _ef_max = atoi(argv[i + 1]);
    else if  (arg == "-ef_search") ef_search_str = string(argv[i+1]);
    else if  (arg == "-index_file") index_file = string(argv[i+1]);
    else if (arg == "-M") M = atoi(argv[i + 1]);
    else if (arg == "-nthreads") thread = atoi(argv[i + 1]);
    else{
      //show error
      cout << "Error: invalid argument:" << arg << endl;
      exit(-1);
    }
  }

  // assert(index_k_list.size() != 0);
  // parse ef_search_str into ef_search_list
  vector<int> ef_search_list;
  {
    std::stringstream ss(ef_search_str);
    int ef_search;
    while (ss >> ef_search) {
      ef_search_list.push_back(ef_search);
      if (ss.peek() == ',') ss.ignore();
    }
  }
  assert(ef_search_list.size() != 0);
  assert(ef_construction != -1);

  // assert(groundtruth_path != "");

  timeval t1, t2;

  DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
  data_wrapper.readData(dataset_path, query_path);
  cout << "get data size:" << data_wrapper.nodes.size() << " query size:" << data_wrapper.querys.size() << endl;
  data_wrapper.nodes_keys = get_label(data_size, label_path);
  cout << "get data label size:" << data_wrapper.nodes_keys.size() << endl;
  data_wrapper.query_ranges = get_range(query_num, qrange_path);
  cout << "get query range size:" << data_wrapper.query_ranges.size() << endl;


  // Generate groundtruth
  // data_wrapper.generateRangeFilteringQueriesAndGroundtruthBenchmark(false);
  // Or you can load groundtruth from the given path
  // data_wrapper.LoadGroundtruth(groundtruth_path);

  //Load groundtruth by new method
  data_wrapper.groundtruth = load_groundtruth(groundtruth_path, data_wrapper.query_ranges.size(), query_k);
  cout << "get groundtruth size:" << data_wrapper.groundtruth.size() << endl;
  
  data_wrapper.query_ids.resize(data_wrapper.querys.size());
  std::iota(data_wrapper.query_ids.begin(), data_wrapper.query_ids.end(), 0);//init ori id
  cout << "get query id size:" << data_wrapper.query_ids.size() << endl;
  cout << "query range size:" << data_wrapper.query_ranges.size() << endl;
  assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());

  gettimeofday(&t1, NULL);
  od2id = sort_data_by_label(data_wrapper.nodes, data_wrapper.groundtruth, data_wrapper.nodes_keys, data_size, query_k);
  cout << "get od2id size:" << od2id.size() << endl;
  label_range_2_id_range(data_wrapper.query_ranges, data_wrapper.nodes_keys);
  gettimeofday(&t2, NULL);
  logTime(t1, t2, "Sort Dataset Time");

  // vector<int> searchef_para_range_list = {16, 64, 256};

  cout << "index K(M?):" << endl;
  cout << M << endl;
  // print_set(index_k_list);
  cout << "ef construction:" << endl;
  cout << ef_construction << endl;
  cout << "ef search:" << endl;
  print_set(ef_search_list);
  cout << "thread num:" << thread << endl;

  ef_max_list[0] = _ef_max;
  cout << "ef max:" << ef_max_list[0] << endl;

  SeRF::serf_omp_set_num_threads(thread);

  data_wrapper.version = version;

  base_hnsw::L2Space ss(data_wrapper.data_dim);


  //默认只有一轮
  // for (unsigned index_k : index_k_list) {//{8} M ?
    for (unsigned ef_max : ef_max_list) {//{500}
      // for (unsigned ef_construction : ef_construction_list) {//{100}
        BaseIndex::IndexParams i_params(M, ef_construction,
                                        ef_construction, ef_max);
        {
          cout << endl;
          i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;
          SeRF::IndexSegmentGraph2D index(&ss, &data_wrapper);
          // rangeindex::RecursionIndex index(&ss, &data_wrapper);
          BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D",
                                            "benchmark");

          cout << "Method: " << search_info.method << endl;
          cout << "parameters: ef_construction ( " +
                      to_string(i_params.ef_construction) + " )  index-k( "
               << i_params.K << ")  ef_max (" << i_params.ef_max << ") "
               << endl;
          //check if index file exists
          std::ifstream file(index_file);
          if(file.good()){
            cout << "load index from " << index_file << endl;
            index.buildIndexFromFile(index_file, &i_params);
            cout << "Total # of Neighbors: " << index.index_info->nodes_amount
                << endl;
          }
          else{
            gettimeofday(&t1, NULL);
            index.buildIndex(&i_params);//construct index
            gettimeofday(&t2, NULL);
            logTime(t1, t2, "Build Index Time");
            cout << "Total # of Neighbors: " << index.index_info->nodes_amount
                << endl;
                
            cout << "save index to " << index_file << endl;
            index.write(index_file);
          }
          {
            timeval tt3, tt4;
            BaseIndex::SearchParams s_params;
            s_params.query_K = data_wrapper.query_k;
            for (auto one_searchef : ef_search_list) {
              cout << "ef search:" << one_searchef << endl;
              s_params.search_ef = one_searchef;
              std::map<int, std::pair<float, float>>
                  result_recorder;  // first->precision, second->query_time
              std::map<int, float> comparison_recorder;
              gettimeofday(&tt3, NULL);
              for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {//each query
                int one_id = data_wrapper.query_ids.at(idx);
                s_params.query_range =
                    data_wrapper.query_ranges.at(idx).second -
                    data_wrapper.query_ranges.at(idx).first + 1;
                auto res = index.rangeFilteringSearchOutBound(
                    &s_params, &search_info, data_wrapper.querys.at(one_id),
                    data_wrapper.query_ranges.at(idx));
                search_info.precision =
                    countPrecision(data_wrapper.groundtruth.at(idx), res);
                result_recorder[0].first +=
                    search_info.precision;
                result_recorder[0].second +=
                    search_info.internal_search_time;
                comparison_recorder[0] +=
                    search_info.total_comparison;
              }
              gettimeofday(&tt4, NULL);
              cout << endl
                   << "Search ef: " << one_searchef << endl
                   << "========================" << endl;
              log_result_recorder(result_recorder, comparison_recorder,
                                  data_wrapper.query_ids.size());
              cout << "========================" << endl;
              auto time_val = CountTime(tt3, tt4);
              cout << "qps: " << data_wrapper.query_ids.size() / time_val << endl;
              logTime(tt3, tt4, "total query time");
            }
          }
        }
      // }
    }
  // }

  return 0;
}