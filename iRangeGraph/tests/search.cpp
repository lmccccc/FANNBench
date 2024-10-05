#include "iRG_search.h"

std::unordered_map<std::string, std::string> paths;

int query_K = 10;
int M;

void Generate(iRangeGraph::DataLoader &storage)
{
    storage.LoadData(paths["data_vector"]);
    iRangeGraph::QueryGenerator generator(storage.data_nb, storage.query_nb);
    generator.GenerateRange(paths["range_saveprefix"]);
    storage.LoadQueryRange(paths["range_saveprefix"]);
    generator.GenerateGroundtruth(paths["groundtruth_saveprefix"], storage);
}

void init()
{
    // data vectors should be sorted by the attribute values in ascending order
    paths["data_vector"] = "";

    paths["query_vector"] = "";
    // the path of document where range files are saved
    paths["range_saveprefix"] = "";
    // the path of document where groundtruth files are saved
    paths["groundtruth_saveprefix"] = "";
    // the path where index file is saved
    paths["index"] = "";
    // the path of document where search result files are saved
    paths["result_saveprefix"] = "";
    // M is the maximum out-degree same as index build
}

std::vector<int> LoadId2Od(std::string filename, int N)
{
    std::vector<int> id2od;
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    if (!infile.is_open())
    {
        throw Exception("cannot open " + filename);
    }
    int size;
    infile.read((char *)&size, sizeof(int));
    assert(N == size);
    id2od.resize(size);
    for (int i = 0; i < size; i++)
    {
        infile.read((char *)&id2od[i], sizeof(int));
    }
    infile.close();
    return id2od;
}

void RedriectQueryRange(iRangeGraph::DataLoader &storage, std::vector<int> &od2id, std::vector<int> &attr, int N, int Nq)
{
    std::vector<std::pair<int, int>> p;
    for (int i = 0; i < N; i++)
    {
        p.push_back({attr[od2id[i]], i});//first: attr value, second: id
    }

    auto& t = storage.query_range[0];
    assert(t.size() == Nq);
    for (int qid = 0; qid < Nq; qid++)
    {
        int ql = t[qid].first;
        int qr = t[qid].second;
        int l_bound = std::lower_bound(p.begin(), p.end(), std::make_pair(ql, -1)) - p.begin();
        int r_bound = std::upper_bound(p.begin(), p.end(), std::make_pair(qr, N)) - p.begin() - 1;
            // std::cout << "ori l_id:" << ql << ", new l_bound:" << l_bound << 
            //     ", ori r_id:" << qr << ", new r_bound:" << r_bound << std::endl;
        storage.query_range[0][qid] = {l_bound, r_bound};
    }
}

void RedirectGroundtruth(iRangeGraph::DataLoader &storage, std::vector<int> &id2od, int Nq, int k)
{
    assert(storage.groundtruth[0].size() == Nq);
    for (int i = 0; i < Nq; i++)
    {
        for(int j = 0; j < k; j++)
        {
            storage.groundtruth[0][i][j] = id2od[storage.groundtruth[0][i][j]];
        }
    }
}

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

void ReorderData(iRangeGraph::iRangeGraph_Search<float> &index, std::vector<int> &id2od, int N)
{
    std::vector<std::vector<float>> vec_tmp;
    vec_tmp.resize(N);
    std::cout << "N:" << N << std::endl;
    std::cout << "id2od size:" << id2od.size() << std::endl;
    std::cout << "data_points size:" << index.storage->data_points.size() << std::endl;
    for (int i = 0; i < N; i++)
    {
        vec_tmp[i] = index.storage->data_points[id2od[i]];
    }
    std::swap(index.storage->data_points, vec_tmp);
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


int main(int argc, char **argv)
{
    // init();
    int N, Nq;
    std::string efs;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--range_saveprefix")
            paths["range_saveprefix"] = argv[i + 1];
        if (arg == "--groundtruth_saveprefix")
            paths["groundtruth_saveprefix"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index"] = argv[i + 1];
        if (arg == "--result_saveprefix")
            paths["result_saveprefix"] = argv[i + 1];
        if (arg == "--id2od_file")
            paths["id2od_file"] = argv[i + 1];
        if (arg == "--attr_file")
            paths["attr_file"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--N")
            N = std::stoi(argv[i + 1]);
        if (arg == "--K")
            query_K = std::stoi(argv[i + 1]);
        if (arg == "--Nq")
            Nq = std::stoi(argv[i + 1]);
        if (arg == "--ef_search")
            efs = argv[i + 1];
    }

    if (argc != 27)
        throw Exception("please check input parameters");

    std::cout << "data file:" << paths["data_vector"] << std::endl;
    std::cout << "query file:" << paths["query_vector"] << std::endl;
    std::cout << "range file:" << paths["range_saveprefix"] << std::endl;
    std::cout << "groundtruth file:" << paths["groundtruth_saveprefix"] << std::endl;
    std::cout << "index file:" << paths["index"] << std::endl;
    std::cout << "result file:" << paths["result_saveprefix"] << std::endl;
    std::cout << "id2od file:" << paths["id2od_file"] << std::endl;
    std::cout << "attr file:" << paths["attr_file"] << std::endl;
    std::cout << "M:" << M << std::endl;
    std::cout << "N:" << N << std::endl;
    std::cout << "Nq:" << Nq << std::endl;
    std::cout << "K:" << query_K << std::endl;
    std::vector<int> SearchEF = splitStringToIntVector(efs);
    std::cout << "ef_search:";
    for (int i = 0; i < SearchEF.size(); i++)
    {
        std::cout << SearchEF[i] << " ";
    }
    std::cout << std::endl;

    iRangeGraph::DataLoader storage;
    storage.query_K = query_K;
    storage.LoadQuery(paths["query_vector"]);
    // If it is the first run, Generate shall be called; otherwise, Generate can be skipped
    // Generate(storage);
    std::vector<int> id2od = LoadId2Od(paths["id2od_file"], N);
    //convert id2od to od2id
    std::vector<int> od2id(N);
    for (int i = 0; i < N; i++) od2id[id2od[i]] = i;
    // std::cout << "load id2od success" << std::endl;
    std::vector<int> attr = LoadAttr(paths["attr_file"], N);
    // std::cout << "load attr success" << std::endl;
    storage.LoadQueryRange2(paths["range_saveprefix"], Nq);
    // std::cout << "load query2range success" << std::endl;
    RedriectQueryRange(storage, od2id, attr, N, Nq);
    // std::cout << "query range redirect success" << std::endl;
    storage.LoadGroundtruth2(paths["groundtruth_saveprefix"]);
    // std::cout << "load groundtruth2 success" << std::endl;
    RedirectGroundtruth(storage, id2od, Nq, query_K);
    std::cout << "groundtruth redirect success" << std::endl;

    iRangeGraph::iRangeGraph_Search<float> index(paths["data_vector"], paths["index"], &storage, M, id2od);
    // std::cout << "load index success" << std::endl;
    // ReorderData(index, id2od, N);
    // searchefs can be adjusted
    // std::vector<int> SearchEF = {1700, 1400, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10};
    index.search(SearchEF, paths["result_saveprefix"], M);
}