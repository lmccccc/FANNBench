import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import pandas as pd
import sys
import os

# title = "Paper Dataset dim N query_size label_method query_method distribution label_range query_label_cnt K Threads M serf_M nprobe ef_construction ef_search gamma M_beta alpha L  partition_size_M beamSize split_factor shift_factor final_beam_multiply kgraph_L iter S R B kgraph_M weight_search Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery File Memory"
# title_list = title.split(' ')

algorithms = ["HNSW", "IVFPQ", "Milvus_HNSW", "Milvus_IVFPQ", "ACORN", "DiskANN", "DiskANN_Stitched", "NHQ_kgraph", "NHQ_nsw", "SeRF", "DSG", "WST_opt", "WST_vamana", "iRangeGraph", "UNIFY"]
name_mapping = ["Faiss_HNSW", "Faiss_IVFPQ", "Milvus_HNSW", "Milvus_IVFPQ", "ACORN", "FDiskANN_VG", "FDiskANN_SVG", "NHQ_kgraph", "NHQ_nsw", "SeRF", "DSG", "WST_opt", "WST_vamana", "iRangeGraph", "UNIFY"]
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
dataset_list = ["sift10M", "spacev10m", "redcaps1m", "YTRGB1m"]
d = {"sift10M": 128, "spacev10m": 100, "redcaps1m": 512, "YTRGB1m": 1024}
query_maps = {}

def print_latex(query_maps):
    print("Algorithm & ", end="")
    for dataset in dataset_list:
        print("\\multicolumn{2}{|c|}{", dataset, "}", end="")
        if dataset != dataset_list[-1]:
            print(" & ", end="")
    print("\\\\")

    print("\\cline{2-9}")
    print(" & ", end="")
    for i in range(len(dataset_list)):
        print(" Size & Memory(GB) & Time ", end="")
        if i != len(dataset_list) - 1:
            print(" & ", end="")
        else:
            print("\\\\")

    print("\\hline")
    for idx, algo in enumerate(algorithms):
        out_algo = name_mapping[idx]
        if "_" in out_algo:
            out_algo = out_algo.replace("_", "\\_")
        print(out_algo, " & ", end="")
        for dataset in dataset_list:
            if dataset not in query_maps.keys() or algo not in query_maps[dataset].keys():
                print(" - & - & - ", end="")
                if dataset != dataset_list[-1]:
                    print(" & ", end="")
                continue
            disk_size = int(query_maps[dataset][algo]["IndexSize"])/1024
            if disk_size <= 0:
                disk_size = '-'
            else:
                disk_size = "{:.2f}".format(disk_size)
            # print(disk_size, " & ", end="")
            size = int(query_maps[dataset][algo]["Memory"])/1024
            if size <= 0:
                size = '-'
            else:
                size = "{:.2f}".format(size)
            time = int(query_maps[dataset][algo]["ConstructionTime"])
            if time <= 0:
                time = '-'
            else:
                time = "{:d}".format(time)
            # print(size, " & ", time, end="")

            print("{} & {} & {}".format(disk_size, size, time), end="")
            if dataset != dataset_list[-1]:
                print(" & ", end="")
        print("\\\\")



# main function
if __name__ == "__main__":
    file_path = sys.argv[1]
    dataset = sys.argv[2]
    query_label_cnt = int(sys.argv[3])
    label_range = int(sys.argv[4])
    label_cnt = int(sys.argv[5])
    query_label = int(sys.argv[6])
    distribution = sys.argv[7]
    label_method = sys.argv[8]
    query_method = sys.argv[9]
    

    # print("file ", file_path)
    with open(file_path, 'r') as file:
        data = pd.read_csv(file)
    # print(data)
    # get index size and construction time
    # due to various configurations, we need to find the suitable row. use the latest row as the reference
    
    for index, row in data.iterrows():
        
        if  row["Dataset"] not in dataset_list or \
            (row["distribution"] != "random" and row["distribution"] != "real") or \
            row["label_cnt"] != 1:
            continue
        
        # if query_label_cnt == 1 and label_cnt > 1 and row["query_label"] != query_label:
        #     continue
        algo = row["Paper"]
        dataset = row["Dataset"]
        index_size = row["IndexSize"]
        const_time = row["ConstructionTime"]

        if algo == "SeRF":
            if row["serf_M"] != 8:
                continue
        if "DiskANN" in algo:
            if row["M"] != 40:
                continue

        if row["Recall"] > 0.1 and row["Threads"] == 1 and (row["query_label"] == 6 or row["query_label_cnt"] == 6 or row["query_label_cnt"] == 50000):
            memory = row["Memory"]
            if dataset not in query_maps.keys():
                query_maps[dataset] = {}
            if algo not in query_maps[dataset].keys():
                query_maps[dataset][algo] = {}
            query_maps[dataset][algo]["Memory"] = memory
            continue
        else:
            memory = 0

        if const_time <= 0:
            continue
        if "IVF" in algo and row["partition_size_M"] == d[dataset]:
            print("algo:", algo, " partition_size_M:", row["partition_size_M"], " d:", d[dataset])
            continue
        
        if dataset not in query_maps.keys():
            query_maps[dataset] = {}
        if algo not in query_maps[dataset].keys():
            query_maps[dataset][algo] = {}
        query_maps[dataset][algo]["IndexSize"] = index_size
        query_maps[dataset][algo]["ConstructionTime"] = const_time
        
    
    # print(query_maps)
    print_latex(query_maps)
    
    times = []
    sizes = []
    plt.figure(figsize=(10, 6))
    for dataset in query_maps:
        t_times = []
        t_sizes = []
        for algo in query_maps[dataset]:
            if algo not in algorithms:
                continue
            if "Milvus" in algo or "DiskANN" in algo or "iRange" in algo:
                continue
            if "Memory" not in query_maps[dataset][algo] or "ConstructionTime" not in query_maps[dataset][algo]:
                continue
            size = int(query_maps[dataset][algo]["IndexSize"])
            time = int(query_maps[dataset][algo]["ConstructionTime"])
            # print("algo:", algo)
            times.append(time)
            sizes.append(size)
            t_times.append(time)
            t_sizes.append(size)
        plt.scatter(times, sizes, alpha=0.7, label=dataset)
    plt.xlabel('ConstructionTime')
    plt.ylabel('Memory')
    correlation = np.corrcoef(times, sizes)[0, 1]

    file = "plot/png/correlation.png"
    plt.savefig(file)
    print("save image to ", file)


    print("Pearson correlation:", correlation)




    
    