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


label_algo = [
          "Milvus_HNSW",
          "Milvus_IVFPQ",
          "HNSW",
          "IVFPQ",
          "ACORN",
          "DiskANN", 
          "DiskANN_Stitched",
          "NHQ_kgraph", 
          "NHQ_nsw"]
cpq_label_algo = [
          "HNSW",
          "ACORN",
          "DiskANN", 
          "DiskANN_Stitched",
          "NHQ_kgraph", 
          "NHQ_nsw"]

line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
query_map = {}


# main function
def get_comps_by_recall(algo, sel, recall):
    if algo not in query_map.keys():
        return 0, 2**31-1
    if sel not in query_map[algo].keys():
        return 0, 2**31-1
    
    best_qps = 0
    best_cpq = 2**31-1
    for item in query_map[algo][sel].values():
        if item["Recall"] >= recall-0.005:
            if item["QPS"] > best_qps:
                best_qps = item["QPS"]
            if item["CompsPerQuery"] < best_cpq and item["CompsPerQuery"] > 1:
                best_cpq = item["CompsPerQuery"]
    
    return best_qps, best_cpq

def savedata(data, _file):
    data = pd.DataFrame(data)
    data.to_csv(_file, index=False)
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

    
    
    target_sel_list = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    id2sel = {1:1, 2:0.9, 3:0.8, 4:0.7, 5:0.6, 6:0.5, 7:0.4, 8:0.3, 9:0.2, 10:0.1, 11:0.09, 12:0.08, 13:0.07, 14:0.06, 15:0.05, 16:0.04, 17:0.03, 18:0.02, 19:0.01, 20:0.001}
    sel2id = {}
    for id in id2sel.keys():
        sel2id[id2sel[id]] = id
    range2sel = {100000:1, 90000:0.9, 80000:0.8, 70000:0.7, 60000:0.6, 50000:0.5, 40000:0.4, 30000:0.3, 20000:0.2, 10000:0.1, 
                 9000:0.09, 8000:0.08, 7000:0.07, 6000:0.06, 5000:0.05, 4000:0.04, 3000:0.03, 2000:0.02, 1000:0.01, 100:0.001}
    sel2range = {}
    for _range in range2sel.keys():
        sel2range[range2sel[_range]] = _range
    
    dataset="sift1M"
    distribution = "random"
    label_range = 500
    target_recall = 0.8
    target_id_list = [sel2id[sel] for sel in target_sel_list]


    for index, row in data.iterrows():
        if row["Dataset"] != dataset or \
            distribution != row["distribution"] or \
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
        if algo not in label_algo or row["label_range"] != label_range:
            continue
        # if algo in label_algo and row["label_range"] != label_range2:
        #     continue
        if row["query_label_cnt"] not in target_id_list and row["query_label"] not in target_id_list:
            continue
        if recall <= 0:
            continue
        if recall > 1:
            recall = recall / 100
        if algo not in query_map.keys():
            query_map[algo] = {}
        if row["query_label"] in target_id_list:
            sel = id2sel[row["query_label"]]
        else:
            continue
        if sel not in query_map[algo].keys():
            query_map[algo][sel] = {}
        
        if recall < 0.3:
            continue
        res_turple = {"Recall": recall, "QPS": qps, "CompsPerQuery": comps}
        if algo == "ACORN" or algo == "HNSW" or algo == "iRangeGraph" or algo == "NHQ_nsw" or algo == "SeRF" or algo == "DSG" or algo == "Milvus_HNSW":
            efs = row["ef_search"]
            query_map[algo][sel][efs] = res_turple
        elif algo == "DiskANN" or algo == "DiskANN_Stitched":
            l = row["L"]
            query_map[algo][sel][l] = res_turple
        elif algo == "NHQ_kgraph":
            l_learch = row["ef_search"]
            query_map[algo][sel][l_learch] = res_turple
        elif algo == "WST_opt":
            beam = row["beamSize"]
            mul = row["final_beam_multiply"]
            query_map[algo][sel][beam*mul] = res_turple
        elif algo == "WST_vamana":
            beam = row["beamSize"]
            query_map[algo][sel][beam] = res_turple
        elif algo == "UNIFY" or algo == "UNIFY_hybrid":
            efs = row["ef_search"]
            al = row["AL"]
            query_map[algo][sel][efs*al] = res_turple
        elif algo == "IVFPQ" or algo == "Milvus_IVFPQ":
            nprobe = row["nprobe"]
            query_map[algo][sel][nprobe] = res_turple

    
    # print(query_map["ACORN"])
    columns = ['Group', 'Algorithm_qps', 'Algorithm_cpq', 'selectivity', 'QPS', 'CPQ', 'x']
    # print(query_map["NHQ_kgraph"])
    classic_algo = ["HNSW", "IVFPQ"]
    arbi_algo = ["ACORN"]
    disk_algo = ["DiskANN", "DiskANN_Stitched"]
    nhq_algo = ["NHQ_kgraph", "NHQ_nsw"]
    qps_list = {}
    cpq_list = {}
    for idx, algo in enumerate(label_algo):
        qps_list[algo] = []
        cpq_list[algo] = []
    for sel in target_sel_list:
        for idx, algo in enumerate(label_algo):
            qps, cpq = get_comps_by_recall(algo, sel, target_recall)
            qps_list[algo].append(qps)
            cpq_list[algo].append(cpq)
    
    classic_qps = [0 for i in range(len(target_sel_list))]
    classic_cpq = [2**31-1 for i in range(len(target_sel_list))]
    classic_qps_algo = ["none" for i in range(len(target_sel_list))]
    classic_cpq_algo = ["none" for i in range(len(target_sel_list))]

    nhq_best_qps = [0 for i in range(len(target_sel_list))]
    nhq_best_cpq = [2**31-1 for i in range(len(target_sel_list))]
    nhq_best_qps_algo = ["none" for i in range(len(target_sel_list))]
    nhq_best_cpq_algo = ["none" for i in range(len(target_sel_list))]

    disk_best_qps = [0 for i in range(len(target_sel_list))]
    disk_best_cpq = [2**31-1 for i in range(len(target_sel_list))]
    disk_best_qps_algo = ["none" for i in range(len(target_sel_list))]
    disk_best_cpq_algo = ["none" for i in range(len(target_sel_list))]

    arbi_best_qps = [0 for i in range(len(target_sel_list))]
    arbi_best_cpq = [2**31-1 for i in range(len(target_sel_list))]
    arbi_best_qps_algo = ["none" for i in range(len(target_sel_list))]
    arbi_best_cpq_algo = ["none" for i in range(len(target_sel_list))]
    
    print("kgraph:", cpq_list["NHQ_kgraph"])
    print("nhq:", cpq_list["NHQ_nsw"])
    for i in range(len(target_sel_list)):
        for algo in label_algo:
            if cpq_list[algo][i] == 1:
                continue
            if algo in classic_algo:
                if classic_qps[i] < qps_list[algo][i]:
                    classic_qps[i] = qps_list[algo][i]
                    classic_qps_algo[i] = algo
                if classic_cpq[i] > cpq_list[algo][i]:
                    classic_cpq[i] = cpq_list[algo][i]
                    classic_cpq_algo[i] = algo
            elif algo in disk_algo:
                if disk_best_qps[i] < qps_list[algo][i]:
                    disk_best_qps[i] = qps_list[algo][i]
                    disk_best_qps_algo[i] = algo
                if disk_best_cpq[i] > cpq_list[algo][i]:
                    disk_best_cpq[i] = cpq_list[algo][i]
                    disk_best_cpq_algo[i] = algo
            elif algo in nhq_algo:
                if nhq_best_qps[i] < qps_list[algo][i]:
                    nhq_best_qps[i] = qps_list[algo][i]
                    nhq_best_qps_algo[i] = algo
                if nhq_best_cpq[i] > cpq_list[algo][i]:
                    nhq_best_cpq[i] = cpq_list[algo][i]
                    nhq_best_cpq_algo[i] = algo
            elif algo in arbi_algo:
                if arbi_best_qps[i] < qps_list[algo][i]:
                    arbi_best_qps[i] = qps_list[algo][i]
                    arbi_best_qps_algo[i] = algo
                if arbi_best_cpq[i] > cpq_list[algo][i]:
                    arbi_best_cpq[i] = cpq_list[algo][i]
                    arbi_best_cpq_algo[i] = algo
    print("nhq qps algo:", nhq_best_qps_algo)
    print("nhq cpq algo:", nhq_best_cpq_algo)
    df = pd.DataFrame(columns=columns)
    for i in range(len(target_sel_list)):
        if classic_qps_algo[i] == "none":
            continue
        new_row = {"Group": "Baseline", 
                   "Algorithm_qps":classic_qps_algo[i],
                    "Algorithm_cpq":classic_cpq_algo[i], 
                    "selectivity": target_sel_list[i], 
                    "QPS": classic_qps[i], 
                    "CPQ": classic_cpq[i],
                    'x': target_sel_list[i]}
        new_row = pd.DataFrame([new_row])
        df = pd.concat([df, new_row], ignore_index=True)
    for i in range(len(target_sel_list)):
        if disk_best_qps_algo[i] == "none":
            continue
        new_row = {"Group": "Best_FDiskANN", 
                   "Algorithm_qps":disk_best_qps_algo[i], 
                   "Algorithm_cpq":disk_best_cpq_algo[i], 
                   "selectivity": target_sel_list[i], 
                   "QPS": disk_best_qps[i], 
                   "CPQ": disk_best_cpq[i],
                   'x': target_sel_list[i]}
        new_row = pd.DataFrame([new_row])
        df = pd.concat([df, new_row], ignore_index=True)
    for i in range(len(target_sel_list)):
        if nhq_best_qps_algo[i] == "none":
            continue
        new_row = {"Group": "Best_NHQ", 
                   "Algorithm_qps":nhq_best_qps_algo[i], 
                   "Algorithm_cpq":nhq_best_cpq_algo[i], 
                   "selectivity": target_sel_list[i], 
                   "QPS": nhq_best_qps[i], 
                   "CPQ": nhq_best_cpq[i],
                   'x': target_sel_list[i]}
        new_row = pd.DataFrame([new_row])
        df = pd.concat([df, new_row], ignore_index=True)
    for i in range(len(target_sel_list)):
        if arbi_best_qps_algo[i] == "none":
            continue
        new_row = {"Group": "Best_Arbitrary", 
                   "Algorithm_qps":arbi_best_qps_algo[i], 
                   "Algorithm_cpq":arbi_best_cpq_algo[i], 
                   "selectivity": target_sel_list[i], 
                   "QPS": arbi_best_qps[i], 
                   "CPQ": arbi_best_cpq[i],
                   'x': target_sel_list[i]}
        new_row = pd.DataFrame([new_row])
        df = pd.concat([df, new_row], ignore_index=True)        

    plt.figure(figsize=(10, 6))
    line_style = line_styles[0 % len(line_styles)]
    marker = markers[0 % len(markers)]
    plt.plot(target_sel_list, classic_qps, label="baseline", linestyle=line_style, marker=marker)

    line_style = line_styles[1 % len(line_styles)]
    marker = markers[1 % len(markers)]
    plt.plot(target_sel_list, disk_best_qps, label="bestDiskANN", linestyle=line_style, marker=marker)

    line_style = line_styles[2 % len(line_styles)]
    marker = markers[2 % len(markers)]
    plt.plot(target_sel_list, nhq_best_qps, label="bestNHQ", linestyle=line_style, marker=marker)

    line_style = line_styles[3 % len(line_styles)]
    marker = markers[3 % len(markers)]
    plt.plot(target_sel_list, arbi_best_qps, label="bestArbi", linestyle=line_style, marker=marker)


    label = "sel_qps_best_" + str(label_range) + "label_" + str(target_recall) + "recall_" + distribution + "_" + dataset
    file = "plot/png/" + label + ".png"
    plt.title('selectivity/qps at 80% recall in '.format(dataset))
    plt.xlabel('Selectivity')
    plt.ylabel('QPS')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig(file)
    print("save image to ", file)

    # xlsfile = xlspath + label + tail
    # print("best algo: ", best_qps_algo)
    # print("best qps: ", best_qps)
    # savedata(df, xlsfile)
        

    plt.figure(figsize=(10, 6))
    line_style = line_styles[0 % len(line_styles)]
    marker = markers[0 % len(markers)]
    plt.plot(target_sel_list, classic_cpq, label="baseline", linestyle=line_style, marker=marker)

    line_style = line_styles[1 % len(line_styles)]
    marker = markers[1 % len(markers)]
    plt.plot(target_sel_list, disk_best_cpq, label="bestDiskANN", linestyle=line_style, marker=marker)

    line_style = line_styles[2 % len(line_styles)]
    marker = markers[2 % len(markers)]
    plt.plot(target_sel_list, nhq_best_cpq, label="bestNHQ", linestyle=line_style, marker=marker)

    line_style = line_styles[3 % len(line_styles)]
    marker = markers[3 % len(markers)]
    plt.plot(target_sel_list, arbi_best_cpq, label="bestArbi", linestyle=line_style, marker=marker)


    label = "sel_qps_labelbest_" + str(label_range) + "label_" + str(target_recall) + "recall_" + distribution + "_" + dataset
    file = "plot/png/" + label + ".png"
    plt.title('selectivity/cpq at 80% recall in '.format(dataset))
    plt.xlabel('Selectivity')
    plt.ylabel('CPQ')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    label = "sel_cpq_labelbest_" + str(label_range) + "label_" + str(target_recall) + "recall_" + distribution + "_" + dataset
    file = "plot/png/" + label + ".png"
    plt.savefig(file)
    print("save image to ", file)

    xlsfile = xlspath + label + tail
    # print("best algo: ", best_cpq_algo)
    savedata(df, xlsfile)