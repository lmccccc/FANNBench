import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import pandas as pd
import sys
import pandas as pd
import os
import csv

# title = "Paper Dataset dim N query_size label_method query_method distribution label_range query_label_cnt K Threads M serf_M nprobe ef_construction ef_search gamma M_beta alpha L  partition_size_M beamSize split_factor shift_factor final_beam_multiply kgraph_L iter S R B kgraph_M weight_search Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery File Memory"
# title_list = title.split(' ')


cpq_range_query_algo = [
          "HNSW",
          "ACORN",
          "DiskANN", 
          "DiskANN_Stitched",
          "NHQ_kgraph", 
          "NHQ_nsw", 
          "SeRF",
          "WST_vamana",
          "WST_opt",
          "UNIFY",
          "UNIFY_hybrid",
          "DSG",
          "iRangeGraph"]
all_algo = [
          "Milvus_HNSW",
          "Milvus_IVFPQ",
          "HNSW",
          "IVFPQ",
          "ACORN",
          "DiskANN", 
          "DiskANN_Stitched",
          "NHQ_kgraph", 
          "NHQ_nsw", 
          "SeRF",
          "WST_vamana",
          "WST_opt",
          "UNIFY",
          "UNIFY_hybrid",
          "DSG",
          "iRangeGraph"]
range_algo = [
          "Milvus_HNSW",
          "Milvus_IVFPQ",
          "HNSW",
          "IVFPQ",
          "ACORN",
          "SeRF",
          "WST_vamana",
          "WST_opt",
          "UNIFY",
          "UNIFY_hybrid",
          "DSG",
          "iRangeGraph"]
label_algo = [
        #   "Milvus_HNSW",
        #   "Milvus_IVFPQ",
        #   "HNSW",
        #   "IVFPQ",
        #   "ACORN",
          "DiskANN", 
          "DiskANN_Stitched",
          "NHQ_kgraph", 
          "NHQ_nsw"]

line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
query_map = {}


def get_comps_by_recall(algo, sel, dist):
    if algo not in query_map.keys():
        return [], [], []
    if sel not in query_map[algo].keys():
        return [], [], []
    if dist not in query_map[algo][sel].keys():
        return [], [], []

    tmp = []
    for item in query_map[algo][sel][dist].keys():
        tmp.append(query_map[algo][sel][dist][item])
    # print(tmp)
    tmp = sorted(tmp, key=lambda x: x["Recall"])
    
    recall_list = []
    qps_list = []
    cpq_list = []
    for idx, item in enumerate(tmp):
        while len(qps_list) > 0 and item["QPS"] > qps_list[-1]:
            recall_list.pop(-1)
            qps_list.pop(-1)
            cpq_list.pop(-1)
        recall_list.append(item["Recall"])
        qps_list.append(item["QPS"])
        cpq_list.append(item["CompsPerQuery"])
    # if algo == "WST_vamana" and sel == 0.5:
    #     print(tmp)
    #     print(recall_list)
    
    return recall_list, qps_list, cpq_list

def savedata(data, _file):
    data = pd.DataFrame(data)
    data.to_csv(_file, index=False)


    
    # df = pd.DataFrame(data, columns=[0.1 * i for i in range(1, 11)])
    # df.to_excel(xlsfile, index=False)
    print("save file to ", _file)

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
    
    plotpath = "plot/"
    xlspath = "plot/csv/"
    tail = ".csv"
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)
    if not os.path.exists(xlspath):
        os.mkdir(xlspath)
    

    # print("file ", file_path)
    with open(file_path, 'r') as file:
        data = pd.read_csv(file)
    # print(data)
    # get index size and construction time
    # due to various configurations, we need to find the suitable row. use the latest row as the reference

    
    
    target_sel_list = [0.1]
    id2sel = {1:1, 2:0.9, 3:0.8, 4:0.7, 5:0.6, 6:0.5, 7:0.4, 8:0.3, 9:0.2, 10:0.1, 11:0.09, 12:0.08, 13:0.07, 14:0.06, 15:0.05, 16:0.04, 17:0.03, 18:0.02, 19:0.01, 20:0.001}
    sel2id = {}
    for id in id2sel.keys():
        sel2id[id2sel[id]] = id
    range2sel = {100000:1, 90000:0.9, 80000:0.8, 70000:0.7, 60000:0.6, 50000:0.5, 40000:0.4, 30000:0.3, 20000:0.2, 10000:0.1, 
                 9000:0.09, 8000:0.08, 7000:0.07, 6000:0.06, 5000:0.05, 4000:0.04, 3000:0.03, 2000:0.02, 1000:0.01, 100:0.001}
    sel2range = {}
    for _range in range2sel.keys():
        sel2range[range2sel[_range]] = _range
    
    dataset="sift10M"
    distribution_list = ["random", "in_dist", "out_dist"]
    label_range = 100000
    label_range2 = 500
    target_id_list = [sel2id[sel] for sel in target_sel_list]
    target_qrange_list = [sel2range[sel] for sel in target_sel_list]


    for index, row in data.iterrows():
        if row["Dataset"] != dataset or \
            label_cnt != row["label_cnt"] or \
            row["Threads"] != 1:
            continue
        if query_label_cnt == 1 and label_cnt > 1 and row["query_label"] != query_label:
            continue
        query_label_cnt = row["query_label_cnt"]
        recall = row["Recall"]
        qps = row["QPS"]
        comps = row["CompsPerQuery"]
        algo = row["Paper"]
        dist = row["distribution"]
        if algo in range_algo and row["label_range"] != label_range:
            continue
        if algo in label_algo and row["label_range"] != label_range2:
            continue
        if row["query_label_cnt"] not in target_id_list and row["query_label_cnt"] not in target_qrange_list:
            continue
        if recall <= 0:
            continue
        if recall > 1:
            recall = recall / 100
        if algo not in query_map.keys():
            query_map[algo] = {}
        if row["query_label_cnt"] in target_id_list:
            sel = id2sel[row["query_label_cnt"]]
        elif row["query_label_cnt"] in target_qrange_list:
            sel = range2sel[row["query_label_cnt"]]
        else:
            continue
        if sel not in query_map[algo].keys():
            query_map[algo][sel] = {}
        if dist not in query_map[algo][sel].keys():
            query_map[algo][sel][dist] = {}
        if recall < 0.3:
            continue
        res_turple = {"Recall": recall, "QPS": qps, "CompsPerQuery": comps}
        if algo == "ACORN" or algo == "HNSW" or algo == "iRangeGraph" or algo == "NHQ_nsw" or algo == "SeRF" or algo == "DSG" or algo == "Milvus_HNSW":
            efs = row["ef_search"]
            query_map[algo][sel][dist][efs] = res_turple
        elif algo == "DiskANN" or algo == "DiskANN_Stitched":
            l = row["L"]
            query_map[algo][sel][dist][l] = res_turple
        elif algo == "NHQ_kgraph":
            l_learch = row["kgraph_L"]
            query_map[algo][sel][dist][l_learch] = res_turple
        elif algo == "WST_opt":
            beam = row["beamSize"]
            mul = row["final_beam_multiply"]
            query_map[algo][sel][dist][beam*mul] = res_turple
        elif algo == "WST_vamana":
            beam = row["beamSize"]
            query_map[algo][sel][dist][beam] = res_turple
        elif algo == "UNIFY" or algo == "UNIFY_hybrid":
            efs = row["ef_search"]
            al = row["AL"]
            query_map[algo][sel][dist][efs*al] = res_turple
        elif algo == "IVFPQ" or algo == "Milvus_IVFPQ":
            nprobe = row["nprobe"]
            query_map[algo][sel][dist][nprobe] = res_turple

    
    # print(query_map["WST_vamana"][0.5])
    columns = ['Algorithm', 'QPS', 'CPQ']
    # print(query_map)
    for sel in target_sel_list:
        for idx, algo in enumerate(all_algo):
            df = pd.DataFrame(columns=columns)
            plt.figure(figsize=(10, 6))
            for dist in distribution_list:
                recall, qps, cpq = get_comps_by_recall(algo, sel, dist)
                for i in range(len(qps)):
                    row = {"Algorithm": algo+"_"+dist, "QPS": qps[i], "CPQ": cpq[i], "recall": recall[i]}
                    new_row = pd.DataFrame([row])
                    df = pd.concat([df, new_row], ignore_index=True)
                line_style = line_styles[idx % len(line_styles)]
                marker = markers[idx % len(markers)]
                # print("algo ", algo, "recall: ", recall, "qps: ", qps)
                plt.plot(recall, qps, label=algo+"_"+dist, linestyle=line_style, marker=marker)

            plt.xlabel('recall')
            plt.ylabel('QPS')
            plt.yscale('log')
            # plt.xscale('log')
            plt.title('recall/qps with different distributions')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            label = "all_dist_recall_qps_" + algo + str(sel) + "sel_" + dataset
            file = "plot/png/" + label + ".png"
            plt.savefig(file)
            print("save image to ", file)


            label = "all_dist_recall_qps_" + algo + str(sel) + "sel_" + dataset
            xlsfile = xlspath + label + tail
            savedata(df, xlsfile)
