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

algorithms = ["ACORN", "DiskANN", "HNSW", "IVFPQ", "NHQ_kgraph", "NHQ_nsw", "Milvus_IVFPQ", "Milvus_HNSW", "WST_opt", "WST_vamana", 
              "RII", "SeRF", "iRangeGraph", "UNIFY", "UNIFY_hybrid", "DSG"]
range_query_algo = ["Milvus_IVFPQ", 
          "Milvus_HNSW", 
          "IVFPQ",
          "HNSW",
          "ACORN",
          "SeRF",
          "DSG",
          "WST_vamana",
          "WST_opt",
          "UNIFY",
          "UNIFY_hybrid",
          "iRangeGraph"]
cpq_range_query_algo = [
          "HNSW",
          "ACORN",
          "SeRF",
          "DSG",
          "WST_vamana",
          "WST_opt",
          "UNIFY",
          "UNIFY_hybrid",
          "iRangeGraph"]
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
query_map = {}


def get_qps_by_recall(algo, sel_list, target_recall):
    qps_values = []
    if algo not in query_map.keys():
        qps_values.extend([1] * len(sel_list))
        return qps_values
    for sel in sel_list:
        if sel not in query_map[algo].keys():
            qps_values.append(1)
            continue

        tmp = []
        for item in query_map[algo][sel].values():
            tmp.append(item)
        tmp = sorted(tmp, key=lambda x: x["Recall"])
        result = []

        for i in range(len(tmp)):
            if tmp[i]["QPS"] > 1:
                result.append(tmp[i])
                break
        max_qps = 1
        for entry in query_map[algo][sel].values():
            if entry["Recall"] >= target_recall-0.01 and entry["QPS"] > max_qps:
                max_qps = entry["QPS"]
        qps_values.append(max_qps)
    return qps_values

def get_comps_by_recall(algo, sel_list, target_recall):
    comp_values = []
    if algo not in query_map.keys():
        comp_values.extend([1] * len(sel_list))
        return comp_values
    for sel in sel_list:
        if sel not in query_map[algo].keys():
            comp_values.append(1)
            continue
        min_comps = int(1e9)
        tmp = []
        for item in query_map[algo][sel].values():
            tmp.append(item)
        tmp = sorted(tmp, key=lambda x: x["Recall"])
        result = []

        for i in range(len(tmp)):
            if tmp[i]["QPS"] > 1:
                result.append(tmp[i])
                break

        for entry in query_map[algo][sel].values():
            if entry["Recall"] >= target_recall-0.01 and entry["CompsPerQuery"] < min_comps:
                min_comps = entry["CompsPerQuery"]
        if min_comps == int(1e9):
            min_comps = 1
        comp_values.append(min_comps)
    return comp_values

def savedata(data, _file, xlist):
    data['Selectivity'] = xlist
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
    K = int(sys.argv[10])
    
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

    
    
    target_sel_list = [0.001, 0.01, 0.1, 0.5]
    id2sel = {1:1, 2:0.9, 3:0.8, 4:0.7, 5:0.6, 6:0.5, 7:0.4, 8:0.3, 9:0.2, 10:0.1, 11:0.09, 12:0.08, 13:0.07, 14:0.06, 15:0.05, 16:0.04, 17:0.03, 18:0.02, 19:0.01, 20:0.001}
    sel2id = {}
    for id in id2sel.keys():
        sel2id[id2sel[id]] = id
    range2sel = {100000:1, 90000:0.9, 80000:0.8, 70000:0.7, 60000:0.6, 50000:0.5, 40000:0.4, 30000:0.3, 20000:0.2, 10000:0.1, 
                 9000:0.09, 8000:0.08, 7000:0.07, 6000:0.06, 5000:0.05, 4000:0.04, 3000:0.03, 2000:0.02, 1000:0.01, 100:0.001}
    sel2range = {}
    for _range in range2sel.keys():
        sel2range[range2sel[_range]] = _range
    
    # dataset="spacev10m"
    # distribution = "real"
    # label_range = 100000
    target_recall_list = [0.9, 0.95, 0.99]
    target_id_list = [sel2id[sel] for sel in target_sel_list]
    target_qrange_list = [sel2range[sel] for sel in target_sel_list]

    for index, row in data.iterrows():
        if row["Dataset"] != dataset or \
            distribution != row["distribution"] or \
            label_range != row["label_range"] or \
            row["label_cnt"] != 1 or \
            row["Threads"] != 1 or \
            row["K"] != K:
            continue
        if query_label_cnt == 1 and label_cnt > 1 and row["query_label"] != query_label:
            continue
        query_label_cnt = row["query_label_cnt"]
        recall = row["Recall"]
        qps = row["QPS"]
        comps = row["CompsPerQuery"]
        algo = row["Paper"]
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
        else:
            sel = range2sel[row["query_label_cnt"]]

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
            l_learch = row["kgraph_L"]
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
            # if row["partition_size_M"] != "" and row["partition_size_M"] != row["dim"]/2:
            #     continue
            nprobe = row["nprobe"]
            query_map[algo][sel][nprobe] = res_turple            

        # res_turple = {"Recall": recall, "QPS": qps, "CompsPerQuery": comps}
        # query_map[algo][sel].append(res_turple)
    

    
    # plot by selectivity
    # plot 0.9 recall   
    for target_recall in target_recall_list:
        data = {}
        plt.figure(figsize=(10, 6))
        for idx, algo in enumerate(range_query_algo):
            qps_values = get_qps_by_recall(algo, target_sel_list, target_recall)
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            # print("algo:", algo, " qps_values:", qps_values)
            data[algo] = qps_values

        x = range_query_algo
        for idx, sel in enumerate(target_sel_list):
            y = []
            for algo in x:
                if algo not in data.keys():
                    y.append(1)
                else:
                    y.append(data[algo][idx])
            plt.bar(x, y)
            plt.xlabel('Algorithms')
            plt.ylabel('QPS')
            plt.yscale('log')
            plt.title('QPS at {recall} Recall, {sel} Selectivity, and {query_method} Label'.format(recall=target_recall, sel=sel, query_method=str(label_range)+ ' ' + distribution))
            # plt.legend()
            plt.grid(True)
            # plt.tight_layout()
            plt.show()
            label = "bar_qps_" + str(target_recall) + "recall_" + str(sel) + "sel_" + str(label_range) + "label_" + distribution + "_" + dataset + "_K" + str(K)
            file = "plot/" + label + ".png"
            plt.savefig(file)
            # print("save file to ", file)
            # Open the file in write mode
            # writ to csv file
        label = "bar_qps_" + str(target_recall) + "recall_" + str(sel) + "label_" + distribution + "_" + dataset + "_K" + str(K)
        xlsfile = xlspath + label + tail
        # print("algo:", range_query_algo)
        savedata(data, xlsfile, target_sel_list)
    
    
    for target_recall in target_recall_list:
        plt.figure(figsize=(10, 6))
        data = {}
        for idx, algo in enumerate(cpq_range_query_algo):
            comps_values = get_comps_by_recall(algo, target_sel_list, target_recall)
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            data[algo] = comps_values
            # print("algo:", algo, " comps_values:", qps_values)
        x = cpq_range_query_algo
        for idx, sel in enumerate(target_sel_list):
            y = []
            for algo in x:
                if algo not in data.keys():
                    y.append(1)
                else:
                    y.append(data[algo][idx])
            plt.bar(x, y)

            plt.xlabel('Selectivity')
            plt.ylabel('Comparasions per Query')
            plt.yscale('log')
            plt.title('bar_CPQ at {recall} Recall with {query_method} Label'.format(recall=target_recall, query_method=str(label_range)+ ' ' + distribution))
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            label = "bar_cpq_" + str(target_recall) + "recall_" + str(sel) + "sel_" + str(label_range) + "label_" + distribution + "_" + dataset + "_K" + str(K)
            file = "plot/" + label + ".png"
            plt.savefig(file)
            # print("save file to ", file)
            # Open the file in write mode
            # writ to csv file
        label = "bar_cpq_" + str(target_recall) + "recall_" + distribution + "_" + dataset + "_K" + str(K)
        xlsfile = xlspath + label + tail
        savedata(data, xlsfile, target_sel_list)
