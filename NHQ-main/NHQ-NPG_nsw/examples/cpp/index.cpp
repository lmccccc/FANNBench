#include "n2/hnsw.h"

#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <chrono>
#include <omp.h>

using namespace std;

void peak_memory_footprint()
{

    unsigned iPid = (unsigned)getpid();

    std::cout << "PID: " << iPid << std::endl;

    std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
    std::ifstream info(status_file);
    if (!info.is_open())
    {
        std::cout << "memory information open error!" << std::endl;
    }
    std::string tmp;
    while (getline(info, tmp))
    {
        if (tmp.find("Name:") != std::string::npos || tmp.find("VmPeak:") != std::string::npos || tmp.find("VmHWM:") != std::string::npos)
            std::cout << tmp << std::endl;
    }
    info.close();
}

void load_data(char *filename, float *&data, unsigned &num, unsigned &dim)
{ // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];
    // std::cout << "num: " << dim << std::endl;

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)//seems to be .bin file format. example: size, vector1, vector2...
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
}

void load_result_data(char *filename, unsigned *&data, unsigned &num, unsigned &dim)
{ 
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error : " << filename << filename << std::endl;
        return;
    }
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    unsigned fsize = (unsigned)ss;
    num = fsize / (dim + 1) / 4;
    data = new unsigned[num * dim];
    in.seekg(0, std::ios::beg);
    for (unsigned i = 0; i < num; i++)
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
}

void SplitString(const string &s, vector<char> &v, const string &c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (string::npos != pos2)
    {
        v.push_back(atoi(s.substr(pos1, pos2 - pos1).c_str()));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(atoi(s.substr(pos1).c_str()));
}

void SplitString(const string &s, vector<int> &v, const string &c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (string::npos != pos2)
    {
        v.push_back(atoi(s.substr(pos1, pos2 - pos1).c_str()));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(atoi(s.substr(pos1).c_str()));
}

void SplitString(const string &s, vector<float> &v, const string &c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (string::npos != pos2)
    {
        v.push_back(atof(s.substr(pos1, pos2 - pos1).c_str()));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(atof(s.substr(pos1).c_str()));
}

void SplitString(const string &s, vector<string> &v, const string &c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2 - pos1).c_str());

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1).c_str());
}

void load_data_txt(char *filename, unsigned &num, unsigned &dim, std::vector<std::vector<std::string>> &data)
{
    std::string temp;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cout << "open file error : " << filename << std::endl;
        exit(-1);
    }
    getline(file, temp);
    std::vector<int> tmp2;
    SplitString(temp, tmp2, " ");
    num = tmp2[0];
    dim = tmp2[1];
    data.resize(num);
    int groundtruth_count = 0;
    while (getline(file, temp))
    {
        SplitString(temp, data[groundtruth_count], " ");
        groundtruth_count++;
    }
    std::cout << "load " << data.size() << " data" << std::endl;
    file.close();
}

void get_label(int N, string file_name, vector<vector<string>>& label){  //read json file, only one label supported
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
  vector<string> _label;
  while (true) {
    //get head of ifstream  
    if(in.peek()==','){
        in >> c;
        continue;
    }
    else{
        in >> temp;
        _label.push_back(std::to_string(temp));
        label.push_back(_label);
        _label.clear();
        i++;
        if(i == N){
            break;
        }
    }
  }
  std::cout << "label size: " << label.size() << std::endl;

  in.close();
}

int main(int argc, char **argv)
{
    if (argc != 9)
    {
        std::cout << argv[0] << " data_file att_file save_graph save_attributetable MaxM0 efConstruction datasize threadnum"
                  << std::endl;
        exit(-1);
    }
    std::cout << "data file: " << argv[1] << std::endl;
    std::cout << "label file: " << argv[2] << std::endl;
    std::cout << "save index model file: " << argv[3] << std::endl;
    std::cout << "save index index file: " << argv[4] << std::endl;
    std::cout << "MaxM0: " << argv[5] << std::endl;
    std::cout << "efConstruction: " << argv[6] << std::endl;
    std::cout << "data size: " << argv[7] << std::endl;
    std::cout << "thread num: " << argv[8] << std::endl;


    int data_size = atoi(argv[7]);
    char *data_path = argv[1];
    float *data_load = NULL;
    unsigned points_num, dim;
    load_data(data_path, data_load, points_num, dim); //bin file?
    std::cout << "data_size: " << data_size << " from " << data_path << std::endl;
    std::cout << "points num: " << points_num << std::endl;
    assert(points_num == data_size);


    char *label_data_path = argv[2];
    unsigned label_num, label_dim;
    std::vector<std::vector<string>> label_data;
    get_label(data_size, label_data_path, label_data);
    // load_data_txt(label_data_path, label_num, label_dim, label_data);

    //label_data.resize(points_num);
    //for (int i = 1000000; i < points_num; i++)
    //{
    //    label_data[i] = label_data[i - 1000000];
    //}

    n2::Hnsw index(dim, "L2");
    for (int i = 0; i < points_num; i++)
    {
        vector<float> tmp(dim);
        for (int j = 0; j < dim; j++)
        {
            tmp[j] = data_load[i * dim + j];
        }
        index.AddData(tmp);
    }
    for (int i = 0; i < points_num; i++)
    {
        index.AddAllNodeAttributes(label_data[i]);
    }
    char* thread_str = argv[8];
    std::cout << "thread num: " << thread_str << std::endl;

    vector<pair<string, string>> configs = {{"M", argv[5]}, {"MaxM0", argv[5]}, {"NumThread", thread_str}, {"efConstruction", argv[6]}};
    index.SetConfigs(configs);
    peak_memory_footprint();
    std::cout<<"indexing ..."<<std::endl;
    auto s = std::chrono::high_resolution_clock::now();
    index.Fit();
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> b_diff = e - s;
    float time = b_diff.count();
    peak_memory_footprint();
    std::cout << "Build time: " << time << std::endl;
    index.SaveModel(argv[3]);
    index.SaveAttributeTable(argv[4]);

    return 0;
}
