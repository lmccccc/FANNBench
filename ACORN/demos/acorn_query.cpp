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

faiss::IndexACORNFlat read_index(char* filepath){
    faiss::IndexACORNFlat* idx = static_cast<faiss::IndexACORNFlat*>(faiss::read_index(filepath, 0));
    return *idx;
}

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

// create indices for debugging, write indices to file, and get recall stats for all queries
int main(int argc, char *argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "====================\nSTART: running TEST_ACORN for hnsw, sift data --\n" << std::endl; // << nthreads << "cores\n" << std::endl;
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

    int opt;
    {// parse arguments

        if (argc != 15) {
            fprintf(stderr, "Syntax: %s <number vecs> <gamma> <dataset> <M> <M_beta> <dataset file> <query>  <attr>  <qrange> <groundtruth> <k> <index file> <nthreads> <efs>\n", argv[0]);
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
        query_file = argv[9];
        std::cout << "query file: " << query_file << std::endl;
        attr_file = argv[10];
        std::cout << "attr file: " << attr_file << std::endl;
        qrange_file = argv[11];
        std::cout << "qrange file: " << qrange_file << std::endl;
        gt_file = argv[12];
        std::cout << "gt file: " << gt_file << std::endl;
        index_file = argv[13];
        std::cout << "index file: " << index_file << std::endl;
        efs_str = argv[14];
        efs_list = splitStringToIntVector(efs_str);
        std::cout << "ef_search: ";
        for (int num : efs_list) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
        gt_size = k;

    }
    

    faiss::faiss_omp_set_num_threads(nthreads);

    // load metadata(attr)
    n_centroids = gamma;
    std::vector<int> metadata = load_json_to_vector<int>(attr_file);
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
    std::vector<int> aq;
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
        aq = load_qrange(qrange_file);
        // printf("[%.3f s] Loaded %ld %s queries\n", elapsed() - t0, nq, dataset.c_str());
        std::cout << "[ " << elapsed() - t0 << "s ] Loaded " << nq << " " << dataset << " queries" << std::endl;
 
    }
    // nq = 1;
    // int gt_size = 100;
    // if (dataset=="sift1M_test" || dataset=="paper") {
    //     gt_size = 10;
    // } 
    std::vector<faiss::idx_t> gt(gt_size * nq);
    { // load ground truth
        gt = load_gt(dataset, gt_file, gamma, alpha, assignment_type, N);
        // printf("[%.3f s] Loaded ground truth, gt_size: %d\n", elapsed() - t0, gt_size);
        std::cout << "[ " << elapsed() - t0 << "s ] Loaded ground truth, gt_size: " << gt_size << std::endl;
    }

    // create normal (base) and hybrid index
    // printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
    //            elapsed() - t0, d, M, N, gamma);
    std::cout << "[ " << elapsed() - t0 << "s ] Index Params -- d: " << d << ", M: " << M << ", N: " << N << ", gamma: " << gamma << std::endl;
    // ACORN-gamma
    // faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata, M_beta);
    faiss::IndexACORNFlat hybrid_index = read_index(index_file);
    hybrid_index.acorn.efSearch = efs; // default is 16 HybridHNSW.capp
    debug("ACORN index created%s\n", "");

    auto nbsize = hybrid_index.acorn.neighbors.size() * sizeof(int32_t);
    auto offset_size = hybrid_index.acorn.offsets.size() * sizeof(size_t);
    auto level_size = hybrid_index.acorn.levels.size() * sizeof(int);
    faiss::IndexFlatCodes* index_codes = static_cast<faiss::IndexFlatCodes*>(hybrid_index.storage);
    auto data_size = index_codes->codes.size() * sizeof(uint8_t);
    auto mem = (nbsize + offset_size + level_size + data_size) * 1.0 / (1024 * 1024);// MB
    std::cout << "acorn size: " << nbsize << ", memory: " << mem << "MB" << std::endl;

    { // print out stats
        // printf("====================================\n");
        // printf("============ ACORN INDEX =============\n");
        // printf("====================================\n");
        std::cout << "====================================" << std::endl;
        std::cout << "============ ACORN INDEX =============" << std::endl;
        std::cout << "====================================" << std::endl;
        hybrid_index.printStats(false);
       
    }

    
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

    //search start
    for (int _efs : efs_list){
    hybrid_index.acorn.efSearch = _efs; // default is 16 HybridHNSW.capp
    { // searching the hybrid database
        // printf("==================== ACORN INDEX ====================\n");
        // printf("[%.3f s] Searching the %d nearest neighbors "
        //        "of %ld vectors in the index, efsearch %d\n",
        //        elapsed() - t0,
        //        k,
        //        nq,
        //        hybrid_index.acorn.efSearch);
        std::cout << "==================== ACORN INDEX ====================" << std::endl;
        std::cout << "[ " << elapsed() - t0 << "s ] Searching the " << k << " nearest neighbors of " << nq << " vectors in the index, efsearch " << hybrid_index.acorn.efSearch << std::endl;

        std::vector<faiss::idx_t> nns2(k * nq);
        std::vector<float> dis2(k * nq);

        // create filter_ids_map, ie a bitmap of the ids that are in the filter
        std::vector<char> filter_ids_map(nq * N);
        uint64_t sel = 0;
        for (int xq = 0; xq < nq; xq++) {
            for (int xb = 0; xb < N; xb++) {
                bool in_range = (metadata[xb] >= aq[xq*2] && metadata[xb] <= aq[xq*2+1]);
                filter_ids_map[xq * N + xb] = in_range;
                if(in_range) sel++;
            }
        }
        double selectivity = sel * 1.0 / (nq * N);
        std::cout << "selectivity: " << selectivity << std::endl;

        double t1_x = elapsed();
        hybrid_index.search(nq, xq, k, dis2.data(), nns2.data(), filter_ids_map.data()); // TODO change first argument back to nq
        double t2_x = elapsed();

        // printf("[%.3f s] Query results (vector ids, then distances):\n",
        //        elapsed() - t0);
        std::cout << "[ " << elapsed() - t0 << "s ] Query results (vector ids, then distances):" << std::endl;
        int nq_print = std::min(5, (int) nq);
        for (int i = 0; i < nq_print; i++) {
            std::cout << "query " << i << " nn's (" << aq[i] << "): ";
            for (int j = 0; j < k; j++) {
                std::cout << nns2[j + i * k] << " (" << metadata[nns2[j + i * k]] << ") ";
            }
            std::cout << std::endl << "     dis: \t";
            for (int j = 0; j < k; j++) {
                std::cout << dis2[j + i * k] << " ";
            }
            std::cout << std::endl;
        }


        // printf("[%.3f s] *** Query time: %f\n",
        //        elapsed() - t0, t2_x - t1_x);
        std::cout << "[ " << elapsed() - t0 << "s ] *** Query time: " << t2_x - t1_x << std::endl;
               
        std::cout << "qps: " << nq / (t2_x - t1_x) << std::endl;

        std::cout << " *** acorn recall " << std::endl;
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
                    if(nns2[qind * k + top] == gt[qind * gt_size + gtind]) {
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
        std::cout << "efs:" << _efs << std::endl;
        std::cout << "acorn get " << positive_size << 
                     " postive res from " << total_size << 
                     " results, recall@" << k << ":"  << recall << std::endl;
        std::cout << "recall dist(10\% per bucket): ";
        for(int i = 0; i < 11; ++i){
            std::cout << hist[i] * 1.0 / nq << " ";
        }
        std::cout << std::endl;
        


        std::cout << "finished hybrid index examples" << std::endl;
    }



    

    // check here

    {// look at stats
        // const faiss::HybridHNSWStats& stats = index.hnsw_stats;
        const faiss::ACORNStats& stats = faiss::acorn_stats;

        std::cout << "============= ACORN QUERY PROFILING STATS =============" << std::endl;
        // printf("[%.3f s] Timing results for search of k=%d nearest neighbors of nq=%ld vectors in the index\n",
        //        elapsed() - t0,
        //        k,
        //        nq);
        std::cout << "[" << elapsed() - t0 << "s] Timing results for search of k=" << k << " nearest neighbors of nq=" << nq << " vectors in the index" << std::endl;
        std::cout << "n1: " << stats.n1 << std::endl;
        std::cout << "n2: " << stats.n2 << std::endl;
        std::cout << "n3 (number distance comps at level 0): " << stats.n3 << std::endl;
        std::cout << "ndis: " << stats.ndis << std::endl;
        std::cout << "nreorder: " << stats.nreorder << std::endl;
        if(stats.n1)
            // printf("average distance computations per query: %f\n", (float)stats.n3 / stats.n1);
            std::cout << "average distance computations per query: " << (float)stats.n3 / stats.n1 << std::endl;
    
    }

    }
    // efs end
    
    // printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
    std::cout << "[ " << elapsed() - t0 << "s ] -----DONE-----" << std::endl;
}