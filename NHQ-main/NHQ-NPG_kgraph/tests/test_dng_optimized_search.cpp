#include <chrono>

#include "efanna2e/index_random.h"
#include "efanna2e/index_graph.h"
#include "efanna2e/util.h"

using namespace std;

void save_result(char *filename, std::vector<std::vector<unsigned>> &results)
{
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++)
  {
    unsigned GK = (unsigned)results[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void load_result_data(char *filename, unsigned *&data, unsigned &num, unsigned &dim)
{ 
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open())
  {
    std::cout << "open file error : " << filename << std::endl;
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

void SplitString(const std::string &s, std::vector<std::string> &v, const std::string &c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;

  while (std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2 - pos1).c_str());

    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length())
    v.push_back(s.substr(pos1).c_str());
}

void SplitString(const std::string &s, std::vector<int> &v, const std::string &c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;

  while (std::string::npos != pos2)
  {
    v.push_back(atoi(s.substr(pos1, pos2 - pos1).c_str()));

    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length())
    v.push_back(atoi(s.substr(pos1).c_str()));
}

void SplitString(const std::string &s, std::vector<char> &v, const std::string &c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;

  while (std::string::npos != pos2)
  {
    v.push_back(atoi(s.substr(pos1, pos2 - pos1).c_str()));

    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length())
    v.push_back(atoi(s.substr(pos1).c_str()));
}

void load_data_txt(char *filename, unsigned &num, unsigned &dim, std::vector<std::vector<string>> &data)
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

void load_groundtruth(string file_name, int Nq, int K, unsigned* &data){//json file
//   vector<vector<int>> gt;
    data = new unsigned[Nq * K];
    std::cout << "gt size: " << Nq * K << std::endl;
    std::ifstream in(file_name);
    if (!in.is_open()) {
        std::cerr << "Error: failed to open file " << file_name << std::endl;
        exit(-1);
    }
    unsigned temp;
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
        // tmp_gt.push_back(temp);
        data[i] = temp;
        i++;
        if(i == Nq * K){
            break;
        }
    }
    in.close();
    return;
}

void get_qrange(int N, string file_name, vector<vector<string>>& label， int query_label_cnt){  //read json file, only one label supported
  std::ifstream in(file_name);
  if (!in.is_open()) {
    std::cerr << "Error: failed to open file " << file_name << std::endl;
    exit(-1);
  }
   //get line
   
  int temp;
  char c;
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
        if(_label.size() == query_label_cnt){
            label.push_back(_label);
            _label.clear();
            i++;
            if(i == N){
                break;
            }
        }
    }
  }
  std::cout << "qrange size: " << label.size() << std::endl;

  in.close();
}

int main(int argc, char **argv)
{
  if (argc != 10)
  {
    std::cout << argv[0] << " graph_path attributetable_path data_path query_path query_att_path groundtruth_path"
              << std::endl;
    exit(-1);
  }

  unsigned seed = 161803398;
  srand(seed);
  //std::cerr << "Using Seed " << seed << std::endl;

  std::cout << "graph file: " << argv[1] << std::endl;
  std::cout << "attributetable file: " << argv[2] << std::endl;
  std::cout << "data file: " << argv[3] << std::endl;
  std::cout << "query file: " << argv[4] << std::endl;
  std::cout << "query attr file: " << argv[5] << std::endl;
  std::cout << "groundtruth file: " << argv[6] << std::endl;
  std::cout << "K: " << argv[7] << std::endl;
  std::cout << "weight search: " << argv[8] << std::endl;
  std::cout << "L search: " << argv[9] << std::endl;
  std::cout << "Query label cnt: " << argv[10] << std::endl;

  int K = atoi(argv[7]);

  char *data_path = argv[3];
  //std::cerr << "Data Path: " << data_path << std::endl;
  unsigned points_num, dim;
  float *data_load = nullptr;
  efanna2e::load_data(data_path, data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim);
  char *query_path = argv[4];
  //std::cerr << "Query Path: " << query_path << std::endl;
  unsigned query_num, query_dim;
  float *query_load = nullptr;
  efanna2e::load_data(query_path, query_load, query_num, query_dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);
  std::cout << "query num: " << query_num << " query dim: " << query_dim << std::endl;


  char *groundtruth_file = argv[6];
  char *attributes_query_file = argv[5];
  unsigned *ground_load = nullptr;
  vector<vector<string>> attributes_query;
  unsigned ground_num, ground_dim;
  ground_num = query_num;
  ground_dim = K;
  unsigned attributes_query_num, attributes_query_dim;
  int qlabel_cnt = atoi(argv[10]);
  // load_result_data(groundtruth_file, ground_load, ground_num, ground_dim);
  load_groundtruth(groundtruth_file, ground_num, ground_dim, ground_load);
  get_qrange(query_num, attributes_query_file, attributes_query, qlabel_cnt);
  // load_data_txt(attributes_query_file, attributes_query_num, attributes_query_dim, attributes_query);
  assert(dim == query_dim);

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::FAST_L2,
                             (efanna2e::Index *)(&init_index));

  char *DNG_path = argv[1];
  //std::cerr << "DNG Path: " << DNG_path << std::endl;

  index.Load(DNG_path);
  index.LoadAttributeTable(argv[2]);
  index.OptimizeGraph(data_load);

  unsigned search_k = K;
  float weight_search = atof(argv[8]); // 140000
  int L_search = atoi(argv[9]); // 100
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L_search);
  paras.Set<float>("weight_search", weight_search);

  std::vector<std::vector<unsigned>> res(query_num);
  for (unsigned i = 0; i < query_num; i++)
    res[i].resize(search_k);

  // Warm up
  for (int loop = 0; loop < 3; ++loop)
  {
    for (unsigned i = 0; i < 10; ++i)
    {
      index.SearchWithOptGraph(attributes_query[i], query_load + i * dim, search_k, paras, res[i].data());
    }
  }

  auto s = std::chrono::high_resolution_clock::now();
  int comps[query_num];
  for (unsigned i = 0; i < query_num; i++) comps[i] = 0;
  for (unsigned i = 0; i < query_num; i++)
  {
    index.SearchWithOptGraph(attributes_query[i], query_load + i * dim, search_k, paras, res[i].data(), &comps[i]);
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  int cnt = 0;
  for (unsigned i = 0; i < ground_num; i++)
  {
    for (unsigned j = 0; j < search_k; j++)
    {
      unsigned k = 0;
      for (; k < ground_dim; k++)
      {
        if (res[i][j] == ground_load[i * ground_dim + k])
          break;
      }
      if (k == ground_dim)
        cnt++;
    }
  }
  int act = 0;
  for (unsigned i = 0; i < query_num; i++) act += comps[i];
  float acc = 1 - (float)cnt / (ground_num * search_k);
  std::cerr << "Search Time: " << diff.count() << " " << search_k << "NN accuracy: " << acc << " Distcount: " << act << " Avgdistcount: " << act * 1.0 / query_num << " qps: " << query_num * 1.0 / diff.count() << std::endl;

  peak_memory_footprint();
  return 0;
}
