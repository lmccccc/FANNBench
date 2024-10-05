#include "construction.h"

std::unordered_map<std::string, std::string> paths;

int M;
int ef_construction;
int threads;


std::vector<int> LoadAttr(std::string filename, int data_nb)
{
    std::vector<int> attr;
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    if (!infile.is_open())
    {
        throw Exception("cannot open " + filename);
    }
    for (int i = 0; i < data_nb; i++)
    {
        int val;
        infile.read((char *)&val, sizeof(int));
        attr.emplace_back(val);
    }
    infile.close();
    return attr;
}

//sort sotrage.data_points by attr, return id2order
std::vector<int> Data_Sort_by_Attr(std::vector<int> attr, iRangeGraph::DataLoader &storage, int N)
{
    std::vector<int> id2order;
    std::vector<int> original_id;
    std::vector<std::pair<int, int>> p;
    for (int i = 0; i < N; i++)
    {
        p.push_back({attr[i], i});//first: attr value, second: id
    }
    sort(p.begin(), p.end());
    std::vector<std::vector<float>> vec_tmp;
    vec_tmp.resize(N);
    id2order.resize(N);original_id;
    original_id.resize(N);

    for (int i = 0; i < N; i++)
    {
        int pid = p[i].second;//pid: id, i: order
        id2order[pid] = i;
        original_id[i] = pid;
        vec_tmp[i] = storage.data_points[pid];
    }
    std::swap(storage.data_points, vec_tmp);
    // for (auto t : storage.query_range)
    // {
    //     std::string domain = t.first;
    //     for (int qid = 0; qid < storage.query_nb; qid++)
    //     {
    //         int ql = t.second[qid].attr_constraints[aid].first;
    //         int qr = t.second[qid].attr_constraints[aid].second;
    //         int l_bound = std::lower_bound(p.begin(), p.end(), std::make_pair(ql, -1)) - p.begin();
    //         int r_bound = std::upper_bound(p.begin(), p.end(), std::make_pair(qr, data_nb)) - p.begin() - 1;
    //         storage.mapped_queryrange[domain].emplace_back(l_bound, r_bound);
    //     }
    // }
    std::cout << "sorted data points" << std::endl;
    return id2order;
}

void save_id2od(std::string filename, std::vector<int> id2od){
    CheckPath(filename);
    std::ofstream indexfile(filename, std::ios::out | std::ios::binary);
    if (!indexfile.is_open())
        throw Exception("cannot open " + filename);
    
    //write id2od size into file
    int size = id2od.size();
    indexfile.write((char *)&size, sizeof(int));
    
    for (int i = 0; i < id2od.size(); i++)
    {
        indexfile.write((char *)&id2od[i], sizeof(int));
    }


    std::cout << "save index done" << std::endl;
    indexfile.close();
}

int main(int argc, char **argv)
{
    int N;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index_save"] = argv[i + 1];
        if (arg == "--attr_file")
            paths["attr"] = argv[i + 1];
        if (arg == "--id2od_file")
            paths["id2od_file"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--N")
            N = std::stoi(argv[i + 1]);
        if (arg == "--ef_construction")
            ef_construction = std::stoi(argv[i + 1]);
        if (arg == "--threads")
            threads = std::stoi(argv[i + 1]);
    }

    if (paths["data_vector"] == "")
        throw Exception("data path is empty");
    if (paths["index_save"] == "")
        throw Exception("index path is empty");
    if (M <= 0)
        throw Exception("M should be a positive integer");
    if (ef_construction <= 0)
        throw Exception("ef_construction should be a positive integer");
    if (threads <= 0)
        throw Exception("threads should be a positive integer");

    iRangeGraph::DataLoader storage;
    storage.LoadData(paths["data_vector"]);
    std::vector<int> attr = LoadAttr(paths["attr"], N);
    std::vector<int> id2order = Data_Sort_by_Attr(attr, storage, N);
    iRangeGraph::iRangeGraph_Build<float> index(&storage, M, ef_construction);
    index.max_threads = threads;
    index.buildandsave(paths["index_save"]);
    save_id2od(paths["id2od_file"], id2order);
}