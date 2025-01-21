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
#include <iomanip>

#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "compact_graph.h"
#include "segment_graph_2d.h"
#include "utils.h"

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

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

//convert label range to id range, only support ordered integer label range from 0 to N-1
void label_range_2_id_range(vector<pair<int, int>> &qrange, vector<int> &sorted_label){
  vector<int> label_start, label_end;
  for (size_t i = 0; i < sorted_label.size(); i++)
  {
    if(label_start.size() == 0) label_start.push_back(i);
    else if(sorted_label[i] != sorted_label[i-1]){
      label_end.push_back(i-1);
      label_start.push_back(i);
    }
    if(i == sorted_label.size() - 1){
      label_end.push_back(i);
    }
  }
  assert(label_start.size() == label_end.size());
  for (size_t i = 0; i < qrange.size(); i++)
  {
    qrange[i].first = label_start[qrange[i].first];
    qrange[i].second = label_end[qrange[i].second];
  }
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
    string label_path = "";
    unsigned index_k = 8;
    unsigned ef_max = 500;
    unsigned ef_construction = 100;
    int query_num = 1000;
    int query_k = 10;
    int nthreads = 1;
    vector<int> od2id;//store original id for recall computation
    

    string indexk_str = "";
    string ef_con_str = "";
    string version = "Benchmark";
    string index_dir_path;

    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") dataset = string(argv[i + 1]);
        if (arg == "-N")
            data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path")
            dataset_path = string(argv[i + 1]);
        if (arg == "-index_path")
            index_dir_path = string(argv[i + 1]);
        if (arg == "-method")
            method = string(argv[i + 1]);
        if (arg == "-k")
            index_k = atoi(argv[i + 1]);
        if (arg == "-ef_max")
            ef_max = atoi(argv[i + 1]);
        if (arg == "-ef_construction")
            ef_construction = atoi(argv[i + 1]);
        if (arg == "-label_path")
            label_path = string(argv[i + 1]);
        if (arg == "-nthreads")
            nthreads = atoi(argv[i + 1]);
        if (arg == "-query_path")
            query_path = string(argv[i + 1]);
    }
    std::cout << "dataset: " << dataset << endl;
    std::cout << "data_size: " << data_size << endl;
    std::cout << "dataset_path: " << dataset_path << endl;
    std::cout << "method: " << method << endl;
    std::cout << "index_k: " << index_k << endl;
    std::cout << "ef_max: " << ef_max << endl;
    std::cout << "ef_construction: " << ef_construction << endl;
    std::cout << "label_path: " << label_path << endl;
    std::cout << "nthreads: " << nthreads << endl;
    std::cout << "query_path: " << query_path << endl;
    std::cout << "index_dir_path: " << index_dir_path << endl;

    Compact::set_num_threads(nthreads);

    DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path); // query_path is useless when just building index
    cout << "get data size:" << data_wrapper.nodes.size() << " query size:" << data_wrapper.querys.size() << endl;
    data_wrapper.nodes_keys = get_label(data_size, label_path);
    cout << "get data label size:" << data_wrapper.nodes_keys.size() << endl;
    // data_wrapper.query_ranges = get_range(query_num, qrange_path);
    // cout << "get query range size:" << data_wrapper.query_ranges.size() << endl;

    cout << "index K:" << index_k<< " ef construction: "<<ef_construction<<" ef_max: "<< ef_max<< endl;

    //Load groundtruth by new method
    // data_wrapper.groundtruth = load_groundtruth(groundtruth_path, data_wrapper.query_ranges.size(), query_k);
    // cout << "get groundtruth size:" << data_wrapper.groundtruth.size() << endl;
    
    // data_wrapper.query_ids.resize(data_wrapper.querys.size());
    // std::iota(data_wrapper.query_ids.begin(), data_wrapper.query_ids.end(), 0);//init ori id
    // cout << "get query id size:" << data_wrapper.query_ids.size() << endl;
    // cout << "query range size:" << data_wrapper.query_ranges.size() << endl;
    // assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());

    timeval t1, t2;
    gettimeofday(&t1, NULL);
    od2id = sort_data_by_label(data_wrapper.nodes, data_wrapper.groundtruth, data_wrapper.nodes_keys, data_size, query_k);
    // label_range_2_id_range(data_wrapper.query_ranges, data_wrapper.nodes_keys);
    gettimeofday(&t2, NULL);
    logTime(t1, t2, "Sort Dataset Time");

    data_wrapper.version = version;
    base_hnsw::L2Space ss(data_wrapper.data_dim);
    BaseIndex* index;

    BaseIndex::IndexParams i_params(index_k, ef_construction,
                                    ef_construction, ef_max);
    i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;
    {
        
        if(method == "Seg2D"){
            index = new SeRF::IndexSegmentGraph2D(&ss, &data_wrapper);
        }else{
            index = new Compact::IndexCompactGraph(&ss, &data_wrapper);
        }

        cout << "method: " << method<<" parameters: ef_construction ( " + to_string(i_params.ef_construction) + " )  index-k( "
                << i_params.K << ")  ef_max (" << i_params.ef_max << ") "
                << endl;
        gettimeofday(&t1, NULL);
        index->buildIndex(&i_params);
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "Build Index Time");
        

        string save_path = index_dir_path; // + "/" + method+ "_" + std::to_string(index_k) + "_" + std::to_string(ef_max) + "_" + std::to_string(ef_construction) + ".bin";
        index->save(save_path);
    }


    return 0;
}