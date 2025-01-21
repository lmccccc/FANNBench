#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>


#include <sys/time.h>

#include <faiss/IndexFlat.h>
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


void get_label(int N, std::string file_name, std::vector<std::vector<uint>>& label){  //read json file, only [1,\n2,\n3,\n...]
    //   uint* label = new uint[N];
    std::ifstream in(file_name);
    if (!in.is_open()) {
        std::cerr << "Error: failed to open file " << file_name << std::endl;
        exit(-1);
    }
    int depth = 0;
    int temp;
    int i = 0;
    label.resize(N);
    while(true){
        char c;
        in >> c;
        if(c == '['){
            depth++;
            continue;
        }
        else if(c == ']'){
            depth--;
            if(depth == 1){
                i++;
                continue;
            }
            if(depth == 0){
                break;
            }
        }
        else if(depth == 2 && c >= '0' && c <= '9'){
            in.seekg(-1, std::ios::cur);
            in >> temp;
            label[i].push_back(temp);
        }
        else{
            continue;
        }
    }
    // std::cout << "label example: " << std::endl;
    // for(int i = 0; i < 10; ++i){
    //     for(int j = 0; j < label[i].size(); ++j){
    //         std::cout << label[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    return;
}

uint* get_range(int N, std::string file_name){  //read json file, only [1,\n2,\n3,\n...]
  uint* label = new uint[N];
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
    label[i] = temp;
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
    std::cout << std::endl << std::endl;
    std::cout << "===============================================================================================================\n";
    std::cout << "START: generating groundtruth\n" << std::endl;
    double t0 = elapsed();


    size_t d = 128; // dimension of the vectors to index - will be overwritten by the dimension of the dataset
    int k = 10;
    
    float* dataset;
    std::vector<std::vector<uint>> attr;
    std::string dataset_file; // must be sift1B or sift1M or tripclick
    std::string attr_file;
    std::string output_file;
    std::string query_file;
    std::string qrange_file;
    
    std::string assignment_type = "rand";

    srand(0); // seed for random number generator

    size_t N = 0; // N will be how many we truncate nb from sift1M to

    int opt;
    {// parse arguments
        
        if (argc != 9) {
            std::cout << "argc=" << argc << std::endl;
            fprintf(stderr, "Syntax: %s <number vecs> <dataset> <attr> <query> <queryrange> <output> <k> <d>\n", argv[0]);
            exit(1);
        }

        N = strtoul(argv[1], NULL, 10);
        printf("N: %ld\n", N);

        dataset_file = argv[2];
        printf("dataset: %s\n", dataset_file.c_str());

        attr_file = argv[3];
        printf("attr: %s\n", attr_file.c_str());

        query_file = argv[4];
        printf("query: %s\n", query_file.c_str());

        qrange_file = argv[5];
        printf("qrange: %s\n", qrange_file.c_str());

        output_file = argv[6];
        printf("output: %s\n", output_file.c_str());
        
        k = atoi(argv[7]);
        printf("topk: %d\n", k);
        
        d = atoi(argv[8]);
        printf("dim: %d\n", d);



        std::cout << "start"<< std::endl;
    }


    // load dataset and attr
    size_t datad, datan;
    dataset = read_file(dataset_file.c_str(), &datad, &datan);
    if (d != datad) {
        d = datad;
    }
    printf("[%.3f s] Loaded dataset, size=%ld, dim=%ld\n", 
        elapsed() - t0, datan, datad);
    
    size_t attrd, attrn;
    // attr = (uint*)read_file(attr_file.c_str(), &attrd, &attrn);
    // assert(attrd == 1);
    // assert(attrn == datan);
    get_label(datan, attr_file, attr);
    printf("[%.3f s] Loaded attr, size=%ld\n", 
        elapsed() - t0, attrn);


    size_t nq;
    float* query;
    uint* qrange;
    { // load query vectors and attributes
        printf("[%.3f s] Loading query vectors and attributes\n", elapsed() - t0);

        size_t d2, nq2;
        // xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        bool is_base = 0;
        // load_data(dataset, is_base, &d2, &nq, xq);
        query = read_file(query_file.c_str(), &d2, &nq);
        assert(d == d2);
        
        std::cout << "query vecs data loaded, with dim: " << d2 << ", size=" << nq << std::endl;
        printf("[%.3f s] Loaded query vectors\n", elapsed() - t0);

        // qrange = (uint*)read_file(qrange_file.c_str(), &d2, &nq2);
        qrange = get_range(nq * 2, qrange_file);
        assert(nq == nq2);

        printf("[%.3f s] Loaded %ld queries\n", elapsed() - t0, nq);

    }

    // create normal (base) and hybrid index
    // base flat index
    faiss::IndexFlat index(d); 




    { // populating the database
        std::cout << "====================Vectors====================\n" << std::endl;
        // printf("====================Vectors====================\n");
       
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        nb = datan;
        d2 = datad;
        assert(d == d2 || !"dataset does not dim 128 as expected");

        std::cout << "data loaded, with dim: " << d2 << ", nb=" << nb << std::endl;

        printf("[%.3f s] Indexing database, size %ld*%ld from max %ld\n",
               elapsed() - t0, N, d2, nb);

        // index.add_attr(N, attr);
        index.add(N, dataset);

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

        delete[] dataset;       
    }

    
    printf("==============================================\n");
    printf("====================Search Results====================\n");
    printf("==============================================\n");
    // double t1 = elapsed();
    printf("==============================================\n");
    printf("====================Search====================\n");
    printf("==============================================\n");
    double t1 = elapsed();
    
    { // searching the base database
        printf("====================INDEX====================\n");
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index\n",
               elapsed() - t0,
               k,
               nq);

        std::vector<std::vector<faiss::idx_t>> nns2(nq);
        std::vector<std::vector<float>> dis2(nq);
        for(int i = 0; i < nq; i++) {
            nns2[i].resize(k);
            dis2[i].resize(k);
        }


        std::cout << "nn and dis size: " << nns2.size() << " " << dis2.size() << std::endl;

        uint64_t sel = 0;
        std::vector<faiss::IDSelectorBatch*> sel_list;
        for (int nq_i = 0; nq_i < 1; nq_i++) { // only do one query label
            std::vector<faiss::idx_t> sel_ids;
            for (int xb = 0; xb < N; xb++) {
                bool in_range = false;
                for(int i = 0; i < attr[xb].size(); i++){
                    if(attr[xb][i] == qrange[nq_i * 2]){
                        in_range = true;
                        break;
                    }
                }
                if(in_range) sel_ids.push_back(xb);
            }
            sel += sel_ids.size();
            sel_list.push_back(new faiss::IDSelectorBatch(sel_ids.size(), sel_ids.data()));
        }
        double selectivity = sel * 1.0 / (nq * N);
        // std::cout << "selectivity: " << selectivity << std::endl;

        double t1 = elapsed();
        for(int q_ind = 0; q_ind < nq; q_ind++){
            faiss::SearchParameters params;
            params.sel = sel_list[0];
            float* xq_i = query + q_ind * d;
            index.search(1, xq_i, k, dis2[q_ind].data(), nns2[q_ind].data(), &params);
        }
        double t2 = elapsed();
        for (int i = 0; i < 1; i++) {
            delete sel_list[i];
        }
        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        // take max of 5 and nq
        std::cout << "[ " << elapsed() - t0 << "s ] Query results (vector ids, then distances):" << std::endl;
        int nq_print = std::min(5, (int) nq);
        for (int i = 0; i < nq_print; i++) {
            std::cout << "query " << i << " nn's (" << qrange[i*2] << "): ";
            for (int j = 0; j < k; j++) {
                std::cout << nns2[i][j] << " (" ;
                for (int attr_idx = 0; attr_idx < attr[nns2[i][j]].size(); ++attr_idx){
                    std::cout << attr[nns2[i][j]][attr_idx] ;
                    if(attr_idx != attr[nns2[i][j]].size()-1){
                        std::cout << ", ";
                    }
                }
                std::cout << ") ";
            }
            std::cout << std::endl << "     dis: \t";
            for (int j = 0; j < k; j++) {
                std::cout << dis2[i][j] << " ";
            }
            std::cout << std::endl;
        }

        printf("[%.3f s] *** Query time: %f\n",
               elapsed() - t0, t2 - t1);

        // calculate selectivity
        sel = 0;
        double hist[11];
        // double selectivity;
        double total_selectivity = 0;
        for (int i = 0; i < 11; i++) hist[i] = 0;
        for (int i = 0; i < nq; i++) {
            sel = 0;
            for (int j = 0; j < N; j++) {
                for(int attr_idx = 0; attr_idx < attr[j].size(); attr_idx++){
                    if (attr[j][attr_idx] == qrange[i*2]){
                        sel++;
                        break;
                    }
                }
            }
            selectivity = (double) sel / N;
            int bucket = selectivity * 10;
            total_selectivity += selectivity;
            hist[bucket]++;
        }
        for (int i = 0; i < 11; i++) hist[i] /= nq;
        total_selectivity /= nq;

        std::cout << "qps:" << nq / (t2 - t1) << std::endl;
        std::cout << "cmp per query:" << N << std::endl;
        std::cout << "recall: 1 of course" << std::endl;
        std::cout << "sel distribution: ";
        for (int i = 0; i < 11; ++i) std::cout << hist[i] << " ";
        std::cout << std::endl;
        std::cout << "selectivity: " << total_selectivity << std::endl;
        
        // print number of distance computations
        // printf("[%.3f s] *** Number of distance computations: %ld\n",
            //    elapsed() - t0, base_index.ntotal * nq);
        std::cout << "finished base index examples" << std::endl;

        //write to file
        std::cout << "====================WRITING====================\n";
        // gt_write(output_file.c_str(), nns.data(), k, nq);
        std::vector<faiss::idx_t> nns;
        for(int i = 0; i < nns2.size(); ++i){
            nns.insert(nns.end(), nns2[i].begin(), nns2[i].end());
        }
        gt_write_json(output_file.c_str(), nns.data(), k, nq);
        std::cout << "==================WRITING DONE=================\n";

        delete[] query, qrange;
    }

    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
}