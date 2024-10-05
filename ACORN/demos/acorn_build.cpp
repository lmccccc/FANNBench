#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>

#include <iomanip>
#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexACORN.h>
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
#include <nlohmann/json.hpp>
#include "utils.cpp"




// create indices for debugging, write indices to file, and get recall stats for all queries
int main(int argc, char *argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "====================\nSTART: running ACORN_INDEX_CONSTRUCTION for hnsw, sift data --\n" << std::endl; // << nthreads << "cores\n" << std::endl;
    // printf("====================\nSTART: running MAKE_INDICES for hnsw --...\n");
    double t0 = elapsed();
    
    int efs = 16; //  default is 16
    std::string efs_str;
    std::vector<int> efs_list;
    int k = 10; // search parameter
    size_t d = 128; // dimension of the vectors to index - will be overwritten by the dimension of the dataset
    int M; // HSNW param M TODO change M back
    int M_beta; // param for compression
    // float attr_sel = 0.001;
    // int gamma = (int) 1 / attr_sel;
    int gamma;
    int n_centroids;
    // int filter = 0;
    std::string dataset; // must be sift1B or sift1M or tripclick
    int test_partitions = 0;
    int step = 10; //2
    
    int alpha = 0;

    srand(0); // seed for random number generator
    int num_trials = 60;


    size_t N = 0; // N will be how many we truncate nb from sift1M to

    char* dataset_file;
    char* attr_file;
    char* index_file;

    int opt;
    {// parse arguments

        if (argc != 11) {
            std::cout << "argc: " << argc << std::endl;
            fprintf(stderr, "Syntax: %s <number vecs> <gamma> <dataset> <M> <M_beta> <dataset> <attr> <index_file> <nthreads>>\n", argv[0]);
            exit(1);
        }

        dataset = argv[1];
        std::cout << "dataset: " << dataset << std::endl;
        N = strtoul(argv[2], NULL, 10);
        std::cout << "N: " << N << std::endl;
        gamma = atoi(argv[3]);
        std::cout << "gamma: " << gamma << std::endl;
        M = atoi(argv[4]);
        std::cout << "M: " << M << std::endl;
        M_beta = atoi(argv[5]);
        std::cout << "M_beta: " << M_beta << std::endl;
        k = atoi(argv[6]);
        std::cout << "topk: " << k << std::endl;
        nthreads = atoi(argv[7]);
        std::cout << "threads: " << nthreads << std::endl;
        dataset_file = argv[8];
        std::cout << "dataset file: " << dataset_file << std::endl;
        attr_file = argv[9];
        std::cout << "attr file: " << attr_file << std::endl;
        index_file = argv[10];
        std::cout << "index file: " << index_file << std::endl;
    }
    
    faiss::faiss_omp_set_num_threads(nthreads);



    // load metadata(attr)
    n_centroids = gamma;
    std::vector<int> metadata = load_json_to_vector<int>(attr_file);
    // printf("loaded base attributes, size: %ld\n", metadata.size());
    std::cout << "loaded base attributes, size:" << metadata.size() << std::endl;
    // std::vector<int> metadata = load_ab(dataset, gamma, assignment_type, N);
    metadata.resize(N);
    assert(N == metadata.size());
    // printf("[%.3f s] Loaded attributions, %ld found\n", 
    //     elapsed() - t0, metadata.size());
    std::cout << "[ " << elapsed() - t0 << "s ] Loaded attributions, " << metadata.size() << " found" << std::endl;

   

    size_t nq;
    float* xq;
    std::vector<int> aq;

    // create normal (base) and hybrid index
    // printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
    //            elapsed() - t0, d, M, N, gamma);
    std::cout << "[ " << elapsed() - t0 << "s ] Index Params -- d: " << d << ", M: " << M << ", N: " << N << ", gamma: " << gamma << std::endl;
    // ACORN-gamma
    faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata, M_beta);
    hybrid_index.acorn.efSearch = efs; // default is 16 HybridHNSW.capp
    debug("ACORN index created%s\n", "");


    { // populating the database
        std::cout << "====================Vectors====================\n" << std::endl;
        // printf("====================Vectors====================\n");
       
        // printf("[%.3f s] Loading database\n", elapsed() - t0);
        std::cout << "[ " << elapsed() - t0 << "s ] Loading database" << std::endl;

        size_t nb, d2;
        bool is_base = 1;
        // std::string filename = get_file_name(dataset, is_base);
        float* xb = fvecs_read(dataset_file, &d2, &nb);
        assert(d == d2 || !"dataset does not dim 128 as expected");
        // printf("[%.3f s] Loaded base vectors from file: %s\n", elapsed() - t0, dataset_file);
        std::cout << "[ " << elapsed() - t0 << "s ] Loaded base vectors from file: " << dataset_file << std::endl;
       

        std::cout << "data loaded, with dim: " << d2 << ", nb=" << nb << std::endl;

        // printf("[%.3f s] Indexing database, size %ld*%ld from max %ld\n",
        //        elapsed() - t0, N, d2, nb);
        std::cout << "[ " << elapsed() - t0 << "s ] Indexing database, size " << N << "*" << d2 << " from max " << nb << std::endl;

        // index->add(nb, xb);

        // printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);
        std::cout << "[ " << elapsed() - t0 << "s ] Adding the vectors to the index" << std::endl;

        hybrid_index.add(N, xb);
        // printf("[%.3f s] Vectors added to hybrid index \n", elapsed() - t0);
        std::cout << "[ " << elapsed() - t0 << "s ] Vectors added to hybrid index" << nb << std::endl;
        std::cout << "Hybrid index vectors added:" << nb << std::endl;
        // printf("SKIPPED creating ACORN-gamma\n");


        delete[] xb;       
    }
   

    // write hybrid index and partition indices to files
    {
        std::cout << "====================Write Index================\n" << std::endl;
        // write hybrid index
        write_index(&hybrid_index, index_file);
        // std::cout << "[" << elapsed() - t0 << "s ] Wrote hybrid index to file: " << filepath << std::endl;
        // printf("[%.3f s] Wrote hybrid index to file: %s\n", elapsed() - t0, index_file);
        std::cout << "[ " << elapsed() - t0 << "s ] Wrote hybrid index to file: " << index_file << std::endl;
        
    }

    { // print out stats
        // printf("====================================\n");
        // printf("============ ACORN INDEX =============\n");
        // printf("====================================\n");
        std::cout << "====================================" << std::endl;
        std::cout << "============ ACORN INDEX =============" << std::endl;
        std::cout << "====================================" << std::endl;
        hybrid_index.printStats(false);

        auto nbsize = hybrid_index.acorn.neighbors.size() * sizeof(int32_t);
        auto offset_size = hybrid_index.acorn.offsets.size() * sizeof(size_t);
        auto level_size = hybrid_index.acorn.levels.size() * sizeof(int);
        faiss::IndexFlatCodes* index_codes = static_cast<faiss::IndexFlatCodes*>(hybrid_index.storage);
        auto data_size = index_codes->codes.size() * sizeof(uint8_t);
        auto mem = (nbsize + offset_size + level_size + data_size) * 1.0 / (1024 * 1024);// MB
        std::cout << "acorn size: " << nbsize << ", memory: " << mem << "MB" << std::endl;
    }
    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
    std::cout << "[ " << elapsed() - t0 << "s ] -----DONE-----" << std::endl;
}