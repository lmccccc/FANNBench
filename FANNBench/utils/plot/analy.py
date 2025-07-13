import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import pandas as pd
import sys
import os
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import normalize
sys.path.append("utils") 
from defination import check_dir, check_file, ivecs_read, fvecs_read, bvecs_read, read_attr, read_file, write_attr_json

# title = "Paper Dataset dim N query_size label_method query_method distribution label_range query_label_cnt K Threads M serf_M nprobe ef_construction ef_search gamma M_beta alpha L  partition_size_M beamSize split_factor shift_factor final_beam_multiply kgraph_L iter S R B kgraph_M weight_search Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery File Memory"
# title_list = title.split(' ')

algorithms = ["HNSW", "IVFPQ", "Milvus_HNSW", "Milvus_IVFPQ", "ACORN", "DiskANN", "DiskANN_Stitched", "NHQ_kgraph", "NHQ_nsw", "SeRF", "DSG", "WST_opt", "WST_vamana", "iRangeGraph", "UNIFY"]
name_mapping = ["Faiss_HNSW", "Faiss_IVFPQ", "Milvus_HNSW", "Milvus_IVFPQ", "ACORN", "FDiskANN_VG", "FDiskANN_SVG", "NHQ_kgraph", "NHQ_nsw", "SeRF", "DSG", "WST_opt", "WST_vamana", "iRangeGraph", "UNIFY"]
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
dataset_list = ["sift10M", "spacev10m", "redcaps1m", "YTRGB1m"]
query_maps = {}

def calculate_js_divergence(vec_set_1, vec_set_2, bins=50):
    """
    计算两个高维向量集合之间的JS散度。
    
    参数:
    - vec_set_1: vector set (numpy array)
    - vec_set_2: vector set (numpy array)
    - bins: size of bin
    
    返回:
    - JS Divergence
    """
    js_divergences = []
    for dim in range(vec_set_1.shape[1]):
        hist_1, bin_edges = np.histogram(vec_set_1[:, dim], bins=bins, density=True)
        hist_2, _ = np.histogram(vec_set_2[:, dim], bins=bin_edges, density=True)
        
        hist_1 = normalize([hist_1], norm='l1')[0]
        hist_2 = normalize([hist_2], norm='l1')[0]
        
        js_div_dim = jensenshannon(hist_1, hist_2)
        js_divergences.append(js_div_dim)
    
    # average value
    return np.mean(js_divergences)

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
    dataset_name = sys.argv[2]
    query_label_cnt = int(sys.argv[3])
    label_range = int(sys.argv[4])
    label_cnt = int(sys.argv[5])
    query_label = int(sys.argv[6])
    distribution = sys.argv[7]
    label_method = sys.argv[8]
    query_method = sys.argv[9]
    dataset_attr_file = sys.argv[10]
    query_range_file = sys.argv[11]
    ground_truth_file = sys.argv[12]
    dataset_file = sys.argv[13]
    query_file = sys.argv[14]
    K = int(sys.argv[15])
    N = int(sys.argv[16])
    
    dataset = read_file(dataset_file)

    query = read_file(query_file)
    Nq, _d = query.shape
    print("query d:", _d)
    print("get query szie:", Nq)  
    
    qrange = read_attr(query_range_file)
    qrange = qrange.reshape(-1, 2)
    print("qrange shape:", qrange.shape)
    _Nq = len(qrange)
    print("get range cnt:", _Nq)
    assert(_Nq == Nq)

    gt = read_attr(ground_truth_file)
    gt = gt.reshape(-1, K)
    Ngt = gt.shape[0]

    attr = read_attr(dataset_attr_file)
    print("attr shape:", attr.shape)
    _N = len(attr)
    print("_N:", _N, " N:", N)
    assert(_N == N)
    
    print("dataset name:", dataset_name, " query label cnt:", query_label_cnt)
    # print("ngt=", Ngt, " nq=", Nq, " k=", K)
    # assert(Ngt == Nq)

    # random sample 10000 dataset pairs (i, j)
    point_szie = 10000
    idx_i = np.random.choice(N, point_szie, replace=False)
    idx_j = np.random.choice(N, point_szie, replace=False)
    dist = np.linalg.norm(dataset[idx_i] - dataset[idx_j], axis=1)  # size * dim
    avg_dist = np.mean(dist)
    print("avg dist:", avg_dist)
    # print("dist shape:", dist.shape)
    # if label_range == 500:
    #     attr_dis = (attr[idx_i] == attr[idx_j]).astype(np.float32)
    # else:
    #     attr_dis = np.abs(attr[idx_i] - attr[idx_j])
    # print("attr_dis shape:", attr_dis.shape)

    # # compute correlation between dis and attr_dis
    # # print("dist[:10]:", dist[:10])
    # # print("attr_dis[:10]:", attr_dis[:10])
    # correlation = np.corrcoef(dist, attr_dis)[0, 1]
    # print("correlation:", correlation, " for dataset:", dataset_name, ", label:", label_range)

    dis = np.zeros((Nq, K), dtype=np.float32)
    for i in range(Nq):
        gt_ids = gt[i]  # K
        t = dataset[gt_ids] - query[i] # K * dim
        dis[i] = np.linalg.norm(t, axis=1) # K
    gt_avg_dis = np.mean(dis)
    dis_pct = gt_avg_dis / avg_dist
    print("gt avg dis:", gt_avg_dis)
    print("gt dis/avg_dis:", dis_pct)

    print("start calculating js divergence")
    js_val = []
    dataset_random_sample = dataset[np.random.choice(N, point_szie, replace=False)]

    query_sample_idx = np.random.choice(Nq, 10, replace=False)
    for ind in query_sample_idx:
        condition = (attr>=qrange[ind][0]) & (attr<= qrange[ind][1])
        subset_idx = np.where(condition)[0]
        subset = dataset[subset_idx]
        if len(subset) > point_szie:
            subset_sample_idx = np.random.choice(len(subset), point_szie, replace=False)
            subset = subset[subset_sample_idx]
        res = calculate_js_divergence(dataset_random_sample, dataset, bins=100)
        js_val.append(res)
        print("divergence for query", ind, ":", res)
    js_val = np.array(js_val)
    avg_js = np.mean(js_val)
    print("avg js:", avg_js, " for dataset:", dataset_name, ", label:", label_range, " qrange:", query_label_cnt)








    
    