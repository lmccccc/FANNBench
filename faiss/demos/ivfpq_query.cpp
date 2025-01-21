#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>


#include <iomanip>
#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


// added these
#include <faiss/Index.h>
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <iostream>
#include <sstream>      // for ostringstream
#include <fstream>  
#include <iosfwd>
#include <faiss/impl/platform_macros.h>
#include <assert.h>     /* assert */
#include <thread>
#include <set>
#include <math.h>  
#include <numeric> // for std::accumulate
#include <cmath>   // for std::mean and std::stdev
#include "utils.cpp"


std::vector<int> splitStringToIntVector(const std::string& str) {
    std::vector<int> result;
    std::istringstream stream(str);
    std::string token;

    // Read the string using a custom delimiter
    while (std::getline(stream, token, ',')) {
        std::istringstream subStream(token); // Create a new stream for the token
        int number;

        // Extract numbers based on space delimiter
        while (subStream >> number) {
            result.push_back(number);
        }
    }

    return result;
}

std::vector<int> get_label(int N, std::string file_name){  //read json file, only [1,\n2,\n3,\n...]
    std::vector<int> label;
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

std::vector<std::vector<faiss::idx_t>> load_groundtruth(std::string file_name, int Nq, int K){//json file
    std::vector<std::vector<faiss::idx_t>> gt;
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
    std::vector<faiss::idx_t> tmp_gt;
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

std::vector<std::pair<int, int>> get_range(int N, std::string file_name){  //read text file, one column, one integer each row
  std::vector<int> label;
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

  std::vector<std::pair<int, int>> qrange;
  for(int i = 0; i < N * 2; i+=2){
    qrange.push_back(std::make_pair(label[i], label[i+1]));
  }
  return qrange;
}

// create indices for debugging, write indices to file, and get recall stats for all queries
int main(int argc, char *argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "====================\nSTART: running TEST_IVFPQ, sift data --\n" << std::endl; // << nthreads << "cores\n" << std::endl;
    // printf("====================\nSTART: running MAKE_INDICES --...\n");
    double t0 = elapsed();
    
    int efs = 16; //  default is 16
    std::string efs_str;
    std::vector<int> efs_list;
    int k = 10; // search parameter
    size_t d = 128; // dimension of the vectors to index - will be overwritten by the dimension of the dataset
    // float attr_sel = 0.001;
    // int gamma = (int) 1 / attr_sel;
    int n_centroids;
    // int filter = 0;
    std::string dataset; // must be sift1B or sift1M or tripclick
    int test_partitions = 0;
    int step = 10; //2
    
    std::string assignment_type = "rand";
    int alpha = 0;

    srand(0); // seed for random number generator
    int num_trials = 60;


    size_t N = 0; // N will be how many we truncate nb from sift1M to

    char* dataset_file;
    char* query_file;
    char* attr_file;
    char* qrange_file;
    char* gt_file;
    char* index_file;
    int gt_size;
    int nprobe;

    int opt;
    {// parse arguments

        if (argc != 13) {
            fprintf(stderr, "Syntax: %s <number vecs> <gamma> <dataset> <M> <M_beta> <dataset file> <query>  <attr>  <qrange> <groundtruth> <k> <index file> <nthreads> <efs> <dim>\n", argv[0]);
            exit(1);
        }

        dataset = argv[1];
        std::cout << "dataset: " << dataset << std::endl;
        N = strtoul(argv[2], NULL, 10);
        std::cout << "N: " << N << std::endl;
        k = atoi(argv[3]);
        std::cout << "topk: " << k << std::endl;
        nthreads = atoi(argv[4]);
        std::cout << "threads: " << nthreads << std::endl;
        dataset_file = argv[5];
        std::cout << "dataset file: " << dataset_file << std::endl;
        query_file = argv[6];
        std::cout << "query file: " << query_file << std::endl;
        attr_file = argv[7];
        std::cout << "attr file: " << attr_file << std::endl;
        qrange_file = argv[8];
        std::cout << "qrange file: " << qrange_file << std::endl;
        gt_file = argv[9];
        std::cout << "gt file: " << gt_file << std::endl;
        index_file = argv[10];
        std::cout << "index file: " << index_file << std::endl;
        nprobe = atoi(argv[11]);
        std::cout << "nprobe: " << nprobe << std::endl;
        gt_size = k;
        
        d = atoi(argv[12]);
        std::cout << "dim: " << d << std::endl;

    }
    

    faiss::faiss_omp_set_num_threads(nthreads);


    std::vector<int> metadata = get_label(N, attr_file);
    // printf("loaded base attributes from: %s\n", attr_file);
    std::cout << "loaded base attributes from:" << attr_file << std::endl;
    // std::vector<int> metadata = load_ab(dataset, gamma, assignment_type, N);
    metadata.resize(N);
    assert(N == metadata.size());
    // printf("[%.3f s] Loaded metadata, %ld attr's found\n", 
    //     elapsed() - t0, metadata.size());
    std::cout << "[ " << elapsed() - t0 << "s ] Loaded metadata, " << metadata.size() << " attr's found" << std::endl;

   

    size_t nq;
    float* xq;
    std::vector<std::pair<int,int>> aq;
    { // load query vectors and attributes
        // printf("[%.3f s] Loading query vectors and attributes\n", elapsed() - t0);
        std::cout << "[ " << elapsed() - t0 << "s ] Loading query vectors and attributes" << std::endl;

        size_t d2;
        bool is_base = 0;

        // std::string filename = get_file_name(dataset, is_base);
        // xq = fvecs_read(filename.c_str(), &d2, &nq);
        xq = fvecs_read(query_file, &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as expected 128");
        if (d != d2) {
            d = d2;
        }
        
        std::cout << "query vecs data loaded, with dim: " << d2 << ", nb=" << nq << std::endl;
        // printf("[%.3f s] Loaded query vectors from %s\n", elapsed() - t0, query_file);
        std::cout << "[ " << elapsed() - t0 << "s ] Loaded query vectors from " << query_file << std::endl;
        // aq = load_aq(dataset, n_centroids, alpha, N);
        aq = get_range(nq, qrange_file);
        // printf("[%.3f s] Loaded %ld %s queries\n", elapsed() - t0, nq, dataset.c_str());
        std::cout << "[ " << elapsed() - t0 << "s ] Loaded " << nq << " " << dataset << " queries" << std::endl;
 
    }
    // nq = 1;
    // int gt_size = 100;
    // if (dataset=="sift1M_test" || dataset=="paper") {
    //     gt_size = 10;
    // } 
    // std::vector<faiss::idx_t> gt(gt_size * nq);
    std::vector<std::vector<faiss::idx_t>> gt;
    { // load ground truth
        // gt = load_gt(dataset, gt_file, gamma, alpha, assignment_type, N);
        gt = load_groundtruth(gt_file, nq, k);
        // printf("[%.3f s] Loaded ground truth, gt_size: %d\n", elapsed() - t0, gt_size);
        std::cout << "[ " << elapsed() - t0 << "s ] Loaded ground truth, gt_size: " << gt_size << std::endl;
    }

    // create normal (base) and hybrid index
    // printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
    //            elapsed() - t0, d, M, N, gamma);
    std::cout << "[ " << elapsed() - t0 << "s ] Index Params -- d: " << d << ", N: " << N << std::endl;

    faiss::IndexIVFPQ* index = dynamic_cast<faiss::IndexIVFPQ*>(faiss::read_index(index_file));
    if (!index) {
        std::cerr << "Failed to load index from file: " << index_file << std::endl;
        return 1;
    }
    // int d2;
    // float* data = fvecs_read(dataset_file, &d2, &N);
    // index.add(N, data);

    index->nprobe = nprobe;
    size_t dataset_size = index->ntotal;
    std::cout << " load index size: " << dataset_size << std::endl;
    debug("Index created%s\n", "");


    
    // printf("==============================================\n");
    // printf("====================Search Results====================\n");
    // printf("==============================================\n");
    std::cout << "==============================================\n";
    std::cout << "====================Search Results====================\n";
    std::cout << "==============================================\n";
    // double t1 = elapsed();
    // printf("==============================================\n");
    // printf("====================Search====================\n");
    // printf("==============================================\n");
    std::cout << "==============================================\n";
    std::cout << "====================Search====================\n";
    std::cout << "==============================================\n";
    double t1 = elapsed();

    std::cout << "==================== IVFPQ INDEX ====================" << std::endl;
    std::cout << "[ " << elapsed() - t0 << "s ] Searching the " << k << " nearest neighbors of " << nq << " vectors in the index" << std::endl;

    std::vector<std::vector<faiss::idx_t>> nns2(nq);
    std::vector<std::vector<float>> dis2(nq);
    for(int i = 0; i < nq; i++) {
        nns2[i].resize(k);
        dis2[i].resize(k);
    }

    // create filter_ids_map, ie a bitmap of the ids that are in the filter
    // std::vector<std::vector<char>> filter_ids_map(nq);
    // for (int i = 0; i < nq; i++) {
    //     filter_ids_map[i].resize(N);
    // }
    uint64_t sel = 0;
    std::vector<faiss::IDSelectorBatch*> sel_list;
    for (int nq_i = 0; nq_i < nq; nq_i++) {
        std::vector<faiss::idx_t> sel_ids;
        for (int xb = 0; xb < N; xb++) {
            bool in_range = (metadata[xb] >= aq[nq_i].first && metadata[xb] <= aq[nq_i].second);
            // filter_ids_map[nq_i][xb] = in_range;
            // if(in_range) sel++;
            if(in_range) sel_ids.push_back(xb);
        }
        sel += sel_ids.size();
        sel_list.push_back(new faiss::IDSelectorBatch(sel_ids.size(), sel_ids.data()));
    }
    double selectivity = sel * 1.0 / (nq * N);
    std::cout << "selectivity: " << selectivity << std::endl;

    double t1_x = elapsed();
    for (int i = 0; i < nq; i++) {
        faiss::SearchParametersIVF params;
        params.nprobe = nprobe;
        params.sel = sel_list[i];
        float* xq_i = xq + i * d;
        index->search(1, xq_i, k, dis2[i].data(), nns2[i].data(), &params);
    }
    for (int i = 0; i < nq; i++) {
        delete sel_list[i];
    }
    // hybrid_index.search(nq, xq, k, dis2.data(), nns2.data(), filter_ids_map.data()); // TODO change first argument back to nq
    double t2_x = elapsed();

    // printf("[%.3f s] Query results (vector ids, then distances):\n",
    //        elapsed() - t0);
    std::cout << "[ " << elapsed() - t0 << "s ] Query results (vector ids, then distances):" << std::endl;
    int nq_print = std::min(5, (int) nq);
    for (int i = 0; i < nq_print; i++) {
        std::cout << "query " << i << " nn's (" << aq[i].first << ", " << aq[i].second << "): ";
        for (int j = 0; j < k; j++) {
            std::cout << nns2[i][j] << " (" << metadata[nns2[i][j]] << ") ";
        }
        std::cout << std::endl << "     dis: \t";
        for (int j = 0; j < k; j++) {
            std::cout << dis2[i][j] << " ";
        }
        std::cout << std::endl;
    }


    // printf("[%.3f s] *** Query time: %f\n",
    //        elapsed() - t0, t2_x - t1_x);
    std::cout << "[ " << elapsed() - t0 << "s ] *** Query time: " << t2_x - t1_x << std::endl;
            
    std::cout << "qps: " << nq / (t2_x - t1_x) << std::endl;

    std::cout << " *** ivfpq recall " << std::endl;
    int total_size = nq * k;
    int positive_size = 0;
    int hist[11];
    for (size_t i = 0; i < 11; i++) hist[i] = 0;
    for (size_t qind = 0; qind < nq; qind++)//each query
    {
        int q_cnt = 0;
        for (size_t top = 0; top < k; top++)//top k res
        {
            for (size_t gtind = 0; gtind < gt_size; gtind++)//gt
            {
                if(nns2[qind][top] == gt[qind][gtind]) {
                    positive_size++;
                    q_cnt++;
                }
            }
        }
        double q_recall = q_cnt * 1.0 / k;
        int q_bucket = q_recall * 10;
        hist[q_bucket]++;
    }
    double recall = positive_size * 1.0 / total_size;
    std::cout << "ivfpq get " << positive_size << 
                    " postive res from " << total_size << 
                    " results, recall@" << k << ":"  << recall << std::endl;
    std::cout << "recall dist(10\% per bucket): ";
    for(int i = 0; i < 11; ++i){
        std::cout << hist[i] * 1.0 / nq << " ";
    }
    std::cout << std::endl;
    


    std::cout << "finished ivfpq index examples" << std::endl;
    



    

    
    // efs end
    
    // printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
    std::cout << "[ " << elapsed() - t0 << "s ] -----DONE-----" << std::endl;
}