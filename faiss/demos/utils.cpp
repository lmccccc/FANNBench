#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>


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
// #include <nlohmann/json.hpp>

// using json = nlohmann::json;
// #include <format>
// for convenience
/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
        -> wget -r ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
        -> cd ftp.irisa.fr/local/texmex/corpus
        -> tar -xf sift.tar.gz
        
 * and unzip it to the sudirectory sift1M.
 **/

// MACRO
#define TESTING_DATA_DIR "./testing_data"


#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <zlib.h>
#include <variant>

// using namespace std;






 /*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/


bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}


float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}


template <typename T>
void gt_write(const char* fname, T* data, size_t d, size_t n){
    std::ofstream file(fname, std::ios::out | std::ios::binary);
    if (!file) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    for (size_t i = 0; i < n; i++)
    {
        try{
            // Write the number of dimensions (as an int)
            file.write(reinterpret_cast<const char*>(&d), sizeof(size_t));
            // Write the float data
            file.write(reinterpret_cast<const char*>(data + i * d), d * sizeof(T));
        } catch (const std::exception& e) {
            // Catching standard exceptions
            std::cerr << "Caught an exception: " << e.what() << std::endl;
        } catch (...) {
            // Catching any other type of exception
            std::cerr << "Caught an unknown exception at " << i << " accessing position " << i * d << std::endl;
        }
    }
    
    file.close();
}

void gt_write_json(const char* fname, long* data, size_t k, size_t n){
    std::ofstream outfile(fname, std::ios::out);
    if (!outfile) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    outfile << "[\n";
    for (size_t i = 0; i < n; i++)
    {
        try{
            outfile << '\t';
            for (size_t j = 0; j < k; j++)
            {
                outfile << data[i * k + j];
                if(j < k - 1){
                    outfile << ", ";
                }
            }
            if (i < n - 1) {
                outfile << ",";
            }
            outfile << "\n";
        } catch (const std::exception& e) {
            // Catching standard exceptions
            std::cerr << "Caught an exception: " << e.what() << std::endl;
        } catch (...) {
            // Catching any other type of exception
            std::cerr << "Caught an unknown exception at " << i << " accessing position " << i * k << std::endl;
        }
    }
    outfile << "]";
    
    outfile.close();
}


uint8_t* bvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % (d + 4) == 0 || !"weird file size");
    size_t n = sz / (d + 4);

    *d_out = d;
    *n_out = n;
    uint8_t* x = new uint8_t[n * (d + 4)];
    size_t nr = fread(x, sizeof(uint8_t), n * (d + 4), f);
    assert(nr == n * (d + 4) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + i * (d + 4), d * sizeof(*x));
    fclose(f);
    return x;
}

//readfile, return int*, float*, or uint8_t* based on file extension
// std::variant<int*, float*, uint8_t*> read_file(const char* fname, size_t* d_out, size_t* n_out){
//     std::string file(fname);
//     std::string ivecs = ".ivecs";
//     std::string fvecs = ".fvecs";
//     std::string bvecs = ".bvecs";

//     if (file.find(ivecs) != std::string::npos) {
//         return ivecs_read(fname, d_out, n_out);
//     } else if(file.find(fvecs) != std::string::npos){
//         return fvecs_read(fname, d_out, n_out);
//     } else if(file.find(bvecs) != std::string::npos){
//         return bvecs_read(fname, d_out, n_out);
//     }
//     else{
//         fprintf(stderr, "could not open %s\n", fname);
//         perror("");
//         abort();
//     }
// }

float* read_file(const char* fname, size_t* d_out, size_t* n_out){
    std::string file(fname);
    std::string fvecs = ".fvecs";

    if(file.find(fvecs) != std::string::npos){
        return fvecs_read(fname, d_out, n_out);
    }
    else{
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
}

// class virtual_io {
//     public:
//     virtual std::variant<int*, float*, uint8_t*> read_file(const char* fname, size_t* d_out, size_t* n_out) = 0;
// };

// class ivecs_io: public virtual_io {

//     public:
//     int* read_file(const char* fname, size_t* d_out, size_t* n_out){
//         return ivecs_read(fname, d_out, n_out);
//     }
// };

// class fvecs_io: public virtual_io {
//     public:
//     float* read_file(const char* fname, size_t* d_out, size_t* n_out){
//         return fvecs_read(fname, d_out, n_out);
//     }
// };

// class bvecs_io: public virtual_io {
//     public:
//     uint8_t* read_file(const char* fname, size_t* d_out, size_t* n_out){
//         return bvecs_read(fname, d_out, n_out);
//     }
// };


// virtual_io* cast_file_type(const char* fname){
//     std::string file(fname);
//     std::string ivecs = ".ivecs";
//     std::string fvecs = ".fvecs";
//     std::string bvecs = ".bvecs";

//     if (file.find(ivecs) != std::string::npos) {
//         return new ivecs_io;
//     } else if(file.find(fvecs) != std::string::npos){
//         return new fvecs_io;
//     } else if(file.find(bvecs) != std::string::npos){
//         return new bvecs_io;
//     }
// }


/*******************************************************
 * Added for debugging
 *******************************************************/
const int debugFlag = 1;

void debugTime() {
	if (debugFlag) {
        struct timeval tval;
        gettimeofday(&tval, NULL);
        struct tm *tm_info = localtime(&tval.tv_sec);
        char timeBuff[25] = "";
        strftime(timeBuff, 25, "%H:%M:%S", tm_info);
        char timeBuffWithMilli[50] = "";
        sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
        std::string timestamp(timeBuffWithMilli);
		std::cout << timestamp << std::flush;
    }
}

//needs atleast 2 args always
//  alt debugFlag = 1 // fprintf(stderr, fmt, __VA_ARGS__); 
#define debug(fmt, ...) \
    do { \
        if (debugFlag == 1) { \
            fprintf(stdout, "--" fmt, __VA_ARGS__);\
        } \
        if (debugFlag == 2) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)



double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*******************************************************
 * performance testing helpers
 *******************************************************/
std::pair<float, float> get_mean_and_std(std::vector<float>& times) {
    // compute mean
    float total = 0;
    // for (int num: times) {
    for (int i=0; i < times.size(); i++) {
       // printf("%f, ", times[i]); // for debugging
        total = total + times[i];
    }
    float mean = (total / times.size());

    // compute stdev from variance, using computed mean
    float result = 0;
    for (int i=0; i < times.size(); i++) {
        result = result + (times[i] - mean)*(times[i] - mean);
    }
    float variance = result / (times.size() - 1);
    // for debugging
    // printf("variance: %f\n", variance);

    float std = std::sqrt(variance);

    // return 
    return std::make_pair(mean, std);
}




// ground truth labels @gt, results to evaluate @I with @nq queries, returns @gt_size-Recall@k where gt had max gt_size NN's per query
float compute_recall(std::vector<faiss::idx_t>& gt, int gt_size, std::vector<faiss::idx_t>& I, int nq, int k, int gamma=1) {
    // printf("compute_recall params: gt.size(): %ld, gt_size: %d, I.size(): %ld, nq: %d, k: %d, gamma: %d\n", gt.size(), gt_size, I.size(), nq, k, gamma);
    
    int n_1 = 0, n_10 = 0, n_100 = 0;
    for (int i = 0; i < nq; i++) { // loop over all queries
        // int gt_nn = gt[i * k];
        std::vector<faiss::idx_t>::const_iterator first = gt.begin() + i*gt_size;
        std::vector<faiss::idx_t>::const_iterator last = gt.begin() + i*gt_size + (k / gamma);
        std::vector<faiss::idx_t> gt_nns_tmp(first, last);
        // if (gt_nns_tmp.size() > 10) {
        //     printf("gt_nns size: %ld\n", gt_nns_tmp.size());
        // }
        
        // gt_nns_tmp.resize(k); // truncate if gt_size > k
        std::set<faiss::idx_t> gt_nns(gt_nns_tmp.begin(), gt_nns_tmp.end());
        // if (gt_nns.size() > 10) {
        //     printf("gt_nns size: %ld\n", gt_nns.size());
        // }
        
        
        for (int j = 0; j < k; j++) { // iterate over returned nn results
            if (gt_nns.count(I[i * k + j])!=0) {
            // if (I[i * k + j] == gt_nn) {
                if (j < 1 * gamma)
                    n_1++;
                if (j < 10 * gamma)
                    n_10++;
                if (j < 100 * gamma)
                    n_100++;
            }
        }
    }
    // BASE ACCURACY
    // printf("* Base HNSW accuracy relative to exact search:\n");
    // printf("\tR@1 = %.4f\n", n_1 / float(nq) );
    // printf("\tR@10 = %.4f\n", n_10 / float(nq));
    // printf("\tR@100 = %.4f\n", n_100 / float(nq)); // not sure why this is always same as R@10
    // printf("\t---Results for %ld queries, k=%d, N=%ld, gt_size=%d\n", nq, k, N, gt_size);
    return (n_10 / float(nq));

}


template <typename T>
void log_values(std::string annotation, std::vector<T>& values) {
    std::cout << annotation;
    for (int i = 0; i < values.size(); i++) {
        std::cout << values[i];
        if (i < values.size() - 1) {
            std::cout << ", ";
        }
    } 
    std::cout << std::endl;
}





















