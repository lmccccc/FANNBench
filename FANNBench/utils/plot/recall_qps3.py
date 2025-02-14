import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import pandas as pd
import sys


# title = "Paper Dataset dim N query_size label_method query_method distribution label_range query_label_cnt K Threads M serf_M nprobe ef_construction ef_search gamma M_beta alpha L  partition_size_M beamSize split_factor shift_factor final_beam_multiply kgraph_L iter S R B kgraph_M weight_search Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery File Memory"
# title_list = title.split(' ')

algorithms = ["ACORN", "DiskANN", "HNSW", "IVFPQ", "NHQ_kgraph", "NHQ_nsw", "Milvus_IVFPQ", "Milvus_HNSW", "WST_opt", "Vamana_tree", 
              "RII", "SeRF", "iRangeGraph", "UNIFY", "DSG"]
range_query_algo = ["ACORN", "HNSW", "IVFPQ", "Milvus_IVFPQ", "Milvus_HNSW", "WST_opt", "Vamana_tree", 
              "SeRF", "iRangeGraph", "UNIFY", "DSG"]
comps_algo = ["ACORN", "DiskANN", "HNSW", "NHQ_kgraph", "NHQ_nsw", "WST_opt", "Vamana_tree", 
              "SeRF", "iRangeGraph","UNIFY", "DSG"]
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
query_map = {}


def get_qps_by_dist(dist, query_label_list, target_recall):
    qps_values = []
    if dist not in query_map.keys():
        qps_values.extend([0] * len(query_label_list))
        return qps_values
    for query_label_cnt in query_label_list:
        if query_label_cnt not in query_map[dist].keys():
            qps_values.append(0)
            continue
        max_qps = -1
        for entry in query_map[dist][query_label_cnt]:
            if entry["Recall"] >= target_recall and entry["QPS"] > max_qps:
                max_qps = entry["QPS"]
        qps_values.append(max_qps)
    return qps_values

def get_comps_by_recall(algo, query_label_list, target_recall):
    comp_values = []
    if algo not in query_map.keys():
        comp_values.extend([0] * len(query_label_list))
        return comp_values
    for query_label_cnt in query_label_list:
        if query_label_cnt not in query_map[algo].keys():
            comp_values.append(0)
            continue
        max_comps = -1
        for entry in query_map[algo][query_label_cnt]:
            if entry["Recall"] >= target_recall and entry["CompsPerQuery"] > max_comps:
                max_comps = entry["CompsPerQuery"]
        comp_values.append(max_comps)
    return comp_values

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
    algo = sys.argv[10]
    print("algo:", algo)
    

    # print("file ", file_path)
    with open(file_path, 'r') as file:
        data = pd.read_csv(file)
    # print(data)
    # get index size and construction time
    # due to various configurations, we need to find the suitable row. use the latest row as the reference
    for index, row in data.iterrows():
        if row["Dataset"] != dataset or \
            algo != row["Paper"] or \
            label_range != row["label_range"] or \
            label_cnt != row["label_cnt"] or \
            distribution != row["distribution"] or\
            row["Threads"] != 1:
            continue
        if query_label_cnt == 1 and label_cnt > 1 and row["query_label"] != query_label:
            continue
        query_label_cnt = row["query_label_cnt"]
        recall = row["Recall"]
        qps = row["QPS"]
        comps = row["CompsPerQuery"]
        dist = row["distribution"]
        if recall <= 0:
            continue
        if recall > 1:
            recall = recall / 100
        if dist not in query_map.keys():
            query_map[dist] = {}
        if row["query_label_cnt"] not in query_map[dist].keys():
            query_map[dist][row["query_label_cnt"]] = []

        res_turple = {"Recall": recall, "QPS": qps, "CompsPerQuery": comps}
        query_map[dist][query_label_cnt].append(res_turple)
    

    query_label_list = [1000 * i for i in range(1, 11)]
    # plot by selectivity
    # plot 0.9 recall
    target_recall_list = [0.9, 0.95, 0.99]
    dist_list = ["random", "in_dist", "out_dist"]

    for target_recall in target_recall_list:
        plt.figure(figsize=(10, 6))
        for idx, dist in enumerate(dist_list):
            qps_values = get_qps_by_dist(dist, query_label_list, target_recall)
            print("dist ", dist , " qps_values", qps_values)
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            plt.plot(query_label_list, qps_values, label=dist, linestyle=line_style, marker=marker)

        plt.xlabel('Selectivity')
        plt.ylabel('QPS')
        plt.yscale('log')
        plt.title('QPS at {recall} Recall with all Label Distribution'.format(recall=target_recall))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        label = str(target_recall) + "recall_" + str(label_range) + "label_" + algo + "_" + dataset
        file = "plot/" + label + ".png"
        plt.savefig(file)
        print("save file to ", file)
    
    
