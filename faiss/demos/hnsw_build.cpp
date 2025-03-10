#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>

#include <iomanip>
#include <sys/time.h>


#include <faiss/IndexHNSW.h>
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

// #include <nlohmann/json.hpp>
#include "utils.cpp"



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

// create indices for debugging, write indices to file, and get recall stats for all queries
int main(int argc, char *argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "====================\nSTART: running HNSW_INDEX_CONSTRUCTION, sift data --\n" << std::endl; // << nthreads << "cores\n" << std::endl;
    double t0 = elapsed();
    
    int k = 100; // search parameter
    size_t d = 128; // dimension of the vectors to index - will be overwritten by the dimension of the dataset

    // int filter = 0;
    std::string dataset; // must be sift1B or sift1M or tripclick
    int test_partitions = 0;
    int step = 10; //2
    int alpha = 0;

    srand(0); // seed for random number generator
    int num_trials = 60;

    int M = 32;
    int ef_construction = 40;
    size_t N = 0; // N will be how many we truncate nb from sift1M to

    char* dataset_file;
    char* attr_file;
    char* index_file;
    char* train_file;

    int opt;
    {// parse arguments

        if (argc != 11) {
            std::cout << "argc: " << argc << std::endl;
            fprintf(stderr, "Syntax: %s <number vecs> <N> <K> <threads> <dataset> <attr> <index> <dim> <M> <efc> \n", argv[0]);
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
        attr_file = argv[6];
        std::cout << "attr file: " << attr_file << std::endl;
        index_file = argv[7];
        std::cout << "index file: " << index_file << std::endl;
        d = atoi(argv[8]);
        std::cout << "dim: " << d << std::endl;
        M = atoi(argv[9]);
        std::cout << "M: " << M << std::endl;
        ef_construction = atoi(argv[10]);
        std::cout << "ef_construction: " << ef_construction << std::endl;

    }


    faiss::faiss_omp_set_num_threads(nthreads);



    // // load metadata(attr)
    // // std::vector<int> metadata = load_json_to_vector<int>(attr_file);
    // std::vector<int> metadata = get_label(N, attr_file);
    // // printf("loaded base attributes, size: %ld\n", metadata.size());
    // std::cout << "loaded base attributes, size:" << metadata.size() << std::endl;
    // // std::vector<int> metadata = load_ab(dataset, gamma, assignment_type, N);
    // // metadata.resize(N);
    // assert(N == metadata.size());
    // // printf("[%.3f s] Loaded attributions, %ld found\n", 
    // //     elapsed() - t0, metadata.size());
    // std::cout << "[ " << elapsed() - t0 << "s ] Loaded attributions, " << metadata.size() << " found" << std::endl;

   

    size_t nq;
    float* xq;
    std::vector<int> aq;

    // create normal (base) and hybrid index
    // printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
    //            elapsed() - t0, d, M, N, gamma);
    std::cout << "[ " << elapsed() - t0 << "s ] Index Params -- d: " << d << ", M: " << M << ", N: " << N << std::endl;

    // make the index object and train it
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = ef_construction;

    { // populating the database
        std::cout << "====================Vectors====================\n" << std::endl;
        // printf("====================Vectors====================\n");
       
        // printf("[%.3f s] Loading database\n", elapsed() - t0);
        std::cout << "[ " << elapsed() - t0 << "s ] Loading database" << std::endl;

        size_t d2;
        bool is_base = 1;
        // std::string filename = get_file_name(dataset, is_base);
        float* xb = fvecs_read(dataset_file, &d2, &N);
        assert(d == d2);
        // printf("[%.3f s] Loaded base vectors from file: %s\n", elapsed() - t0, dataset_file);
        std::cout << "[ " << elapsed() - t0 << "s ] Loaded base vectors from file: " << dataset_file << std::endl;
       

        std::cout << "data loaded, with dim: " << d2 << ", N=" << N << std::endl;

        // printf("[%.3f s] Indexing database, size %ld*%ld from max %ld\n",
        //        elapsed() - t0, N, d2, nb);
        std::cout << "[ " << elapsed() - t0 << "s ] Indexing database, size " << N << "*" << d2 << " from max " << N << std::endl;

        // index->add(nb, xb);

        // printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);
        std::cout << "[ " << elapsed() - t0 << "s ] Adding the vectors to the index" << std::endl;

        index.add(N, xb);
        double t2 = elapsed() - t0;
        std::cout << "index construction cost:" << t2 << std::endl;

        std::cout << "[ " << elapsed() - t0 << "s ] Vectors added to  index" << std::endl;
        std::cout << "Index vectors added:" << N << std::endl;

        delete[] xb;       
    }
   

    // write  index and partition indices to files
    {
        std::cout << "====================Write Index================\n" << std::endl;
        // write  index
        index.printStats();
        write_index(&index, index_file);
        // std::cout << "[" << elapsed() - t0 << "s ] Wrote  index to file: " << filepath << std::endl;
        // printf("[%.3f s] Wrote  index to file: %s\n", elapsed() - t0, index_file);
        
        std::cout << "[ " << elapsed() - t0 << "s ] Wrote hnsw index to file: " << index_file << std::endl;
        
    }

    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
    std::cout << "[ " << elapsed() - t0 << "s ] -----DONE-----" << std::endl;
}