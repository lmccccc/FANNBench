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
        print(" Memory & Time ", end="")
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
                print(" - & - ", end="")
                if dataset != dataset_list[-1]:
                    print(" & ", end="")
                continue
            size = int(query_maps[dataset][algo]["Memory"])
            if size <= 0:
                size = '-'
            time = int(query_maps[dataset][algo]["ConstructionTime"])
            if time <= 0:
                time = '-'
            print(size, " & ", time, end="")
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
        
        if dataset not in query_maps.keys():
            query_maps[dataset] = {}
        if algo not in query_maps[dataset].keys():
            query_maps[dataset][algo] = {}
        query_maps[dataset][algo]["IndexSize"] = index_size
        query_maps[dataset][algo]["ConstructionTime"] = const_time
    
    # print(query_maps)
    print_latex(query_maps)
    
    