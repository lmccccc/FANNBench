import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import pandas as pd
import sys


# title = "Paper Dataset dim N query_size label_method query_method distribution label_range query_label_cnt K Threads M serf_M nprobe ef_construction ef_search gamma M_beta alpha L  partition_size_M beamSize split_factor shift_factor final_beam_multiply kgraph_L iter S R B kgraph_M weight_search Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery File Memory"
# title_list = title.split(' ')

algorithms = ["ACORN", "DiskANN", "HNSW", "IVFPQ", "Milvus_IVFPQ", "Milvus_HNSW", "NHQ_kgraph", "NHQ_nsw", 
              "RII", "SeRF", "Vamana_tree", "iRangeGraph", "WST_opt"]
comps_algo = ["ACORN", "DiskANN", "HNSW", "NHQ_kgraph", "NHQ_nsw", 
              "SeRF", "Vamana_tree", "iRangeGraph", "WST_opt"]
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+']
query_map = {}
# main function
if __name__ == "__main__":
    file_path = sys.argv[1]
    dataset = sys.argv[2]
    query_label_cnt = int(sys.argv[3])
    label_range = int(sys.argv[4])
    label_cnt = sys.argv[5]
    query_label = sys.argv[6]
    distribution = sys.argv[7]

    # print("file ", file_path)
    with open(file_path, 'r') as file:
        data = pd.read_csv(file)
    # print(data)
    # get index size and construction time
    # due to various configurations, we need to find the suitable row. use the latest row as the reference
    for index, row in data.iterrows():
        if row["Dataset"] != dataset or \
            query_label_cnt != row["query_label_cnt"] or \
            label_cnt != row["label_cnt"] or \
            distribution != row["distribution"] or\
            row["Threads"] != 1:
            continue
        if query_label_cnt == 1 and row["query_label"] != query_label:
            continue
        recall = row["Recall"]
        qps = row["QPS"]
        comps = row["CompsPerQuery"]
        algo = row["Paper"]
        if recall <= 0:
            continue

        if algo not in query_map.keys():
            query_map[algo] = []

        query_map[algo].append({"Recall": recall, "QPS": qps, "CompsPerQuery": comps})
        
    for algo in algorithms:
        query_map[algo] = sorted(query_map[algo], key=lambda x: x["Recall"])
    # plot size
    # algorithms = list(query_map.keys())
        # Plot Recall vs QPS
    plt.figure(figsize=(10, 6))
    idx = 0
    for idx, algo in enumerate(query_map.keys()):
        recalls = [entry["Recall"] for entry in query_map[algo]]
        qps_values = [entry["QPS"] for entry in query_map[algo]]
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        plt.plot(recalls, qps_values, label=algo, linestyle=line_style, marker=marker)
        # print("algo:", algo)
        # print("recall:", recalls)
        # print("qps_values:", qps_values)

    plt.xlabel('Recall')
    plt.ylabel('QPS')
    plt.yscale('log')
    plt.title('Recall vs QPS for Different Algorithms at {label_method} Label Method'.format(label_method=label_method))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    label = label_method + "_" + distribution + "_" + str(query_label_cnt) + "_" + str(label_range)
    file = "plot/recall_vs_qps_" + label + ".png"
    plt.savefig(file)


    # Plot Recall vs CompsPerQuery
    plt.figure(figsize=(10, 6))
    idx = 0
    for idx, algo in enumerate(query_map.keys()):
        recalls = [entry["Recall"] for entry in query_map[algo]]
        comps_values = [entry["CompsPerQuery"] for entry in query_map[algo]]
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        plt.plot(recalls, comps_values, label=algo, linestyle=line_style, marker=marker)
        # print("algo:", algo)
        # print("recall:", recalls)
        # print("comps_values:", comps_values)

    plt.xlabel('Recall')
    plt.ylabel('Comparisons Per Query')
    plt.yscale('log')
    plt.title('Recall vs Comps for Different Algorithms at {label_method} Label Method'.format(label_method=label_method))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    label = label_method + "_" + distribution + "_" + str(query_label_cnt) + "_" + str(label_range)
    file = "plot/recall_vs_comps_" + label + ".png"
    plt.savefig(file)


    # Plot qps vs CompsPerQuery
    plt.figure(figsize=(10, 6))
    idx = 0
    for idx, algo in enumerate(query_map.keys()):
        qps = [entry["QPS"] for entry in query_map[algo]]
        comps_values = [entry["CompsPerQuery"] for entry in query_map[algo]]
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        plt.plot(qps, comps_values, label=algo, linestyle=line_style, marker=marker)
        # print("algo:", algo)
        # print("qps:", qps)
        # print("comps_values:", comps_values)

    plt.xlabel('QPS')
    plt.ylabel('Comparisons Per Query')
    # plt.yscale('log')
    plt.title('QPS vs Comps for Different Algorithms at {label_method} Label Method'.format(label_method=label_method))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    label = label_method + "_" + distribution + "_" + str(query_label_cnt) + "_" + str(label_range)
    file = "plot/qps_vs_comps_" + label + ".png"
    plt.savefig(file)

    