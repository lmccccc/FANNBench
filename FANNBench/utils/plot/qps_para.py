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

label_query_algo = ["Milvus_IVFPQ", 
          "Milvus_HNSW", 
          "IVFPQ",
          "HNSW",
          "ACORN", 
          "DiskANN",
          "DiskANN_Stitched",
          "NHQ_kgraph",
          "NHQ_nsw"]
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
query_map = {}


def get_param_by_recall(algo, dataset, sel_list, target_recall):
    qps_values = []
    params = []
    if algo not in query_map[dataset].keys():
        params.extend([-1] * len(sel_list))
        return params
    for sel in sel_list:
        if sel not in query_map[dataset][algo].keys():
            qps_values.append(1)
            params.append(-1)
            continue
        if sel < 0.05 and algo == "UNIFY":
            qps_values.append(1)
            params.append(-1)
            continue

        tmp = []
        for item in query_map[dataset][algo][sel].values():
            tmp.append(item)

        tmp = sorted(tmp, key=lambda x: x["Recall"])
        result = []

        start = 0
        for i in range(len(tmp)):
            if tmp[i]["QPS"] > 1:
                result.append(tmp[i])
                start = i+1
                break
        max_qps = 1
        best_param = -1
        for param, entry in zip(query_map[dataset][algo][sel].keys(), query_map[dataset][algo][sel].values()):
            if entry["Recall"] >= target_recall-0.005 and entry["QPS"] > max_qps:
                max_qps = entry["QPS"]
                best_param = param
        qps_values.append(max_qps)
        params.append(best_param)
    return params


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
    print("dataset:", dataset)
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
    
    dataset_list = ["sift10M", "spacev10m", "redcaps1m", "YTRGB1m"]
    # dataset="spacev10m"
    # distribution = "real"
    label_range = 100000
    target_recall_list = [0.9]
    target_recall = 0.9
    target_id_list = [sel2id[sel] for sel in target_sel_list]
    target_qrange_list = [sel2range[sel] for sel in target_sel_list]

    for index, row in data.iterrows():
        if row["Dataset"] not in dataset_list or \
            label_range != row["label_range"] or \
            label_cnt != row["label_cnt"] or \
            row["Threads"] != 1:
            continue
        if query_label_cnt == 1 and label_cnt > 1 and row["query_label"] != query_label:
            continue
        dataset = row["Dataset"]
        query_label_cnt = row["query_label_cnt"]
        recall = row["Recall"]
        qps = row["QPS"]
        comps = row["CompsPerQuery"]
        algo = row["Paper"]
        if row["query_label_cnt"] not in target_id_list and row["query_label_cnt"] not in target_qrange_list:
            continue
        if recall > 1:
            recall = recall / 100
        if recall < target_recall-0.005:
            continue
        if dataset not in query_map.keys():
            query_map[dataset] = {}
        if algo not in query_map[dataset].keys():
            query_map[dataset][algo] = {}
        if row["query_label_cnt"] in target_id_list:
            sel = id2sel[row["query_label_cnt"]]
        else:
            sel = range2sel[row["query_label_cnt"]]

        if sel not in query_map[dataset][algo].keys():
            query_map[dataset][algo][sel] = {}
        
        if recall < 0.3:
            continue
        res_turple = {"Recall": recall, "QPS": qps, "CompsPerQuery": comps}
        if algo == "ACORN" or algo == "HNSW" or algo == "iRangeGraph" or algo == "NHQ_nsw" or algo == "SeRF" or algo == "DSG" or algo == "Milvus_HNSW":
            efs = row["ef_search"]
            query_map[dataset][algo][sel][efs] = res_turple
        elif algo == "DiskANN" or algo == "DiskANN_Stitched":
            l = row["L"]
            query_map[dataset][algo][sel][l] = res_turple
        elif algo == "NHQ_kgraph":
            l_learch = row["kgraph_L"]
            query_map[dataset][algo][sel][l_learch] = res_turple
        elif algo == "WST_opt":
            beam = row["beamSize"]
            mul = row["final_beam_multiply"]
            if beam not in query_map[dataset][algo][sel].keys():
                query_map[dataset][algo][sel][beam] = res_turple
            elif query_map[dataset][algo][sel][beam]["QPS"] < qps:
                query_map[dataset][algo][sel][beam] = res_turple
        elif algo == "WST_vamana":
            beam = row["beamSize"]
            query_map[dataset][algo][sel][beam] = res_turple
        elif algo == "UNIFY" or algo == "UNIFY_hybrid":
            efs = row["ef_search"]
            al = row["AL"]
            if efs not in query_map[dataset][algo][sel].keys():
                query_map[dataset][algo][sel][efs] = res_turple
            elif query_map[dataset][algo][sel][efs]["QPS"] < qps:
                query_map[dataset][algo][sel][efs] = res_turple
        elif algo == "IVFPQ" or algo == "Milvus_IVFPQ":
            # if row["partition_size_M"] != "" and row["partition_size_M"] != row["dim"]/2:
            #     continue
            nprobe = row["nprobe"]
            query_map[dataset][algo][sel][nprobe] = res_turple            

        # res_turple = {"Recall": recall, "QPS": qps, "CompsPerQuery": comps}
        # query_map[dataset][algo][sel].append(res_turple)
    

    
    # plot by selectivity
    # plot 0.9 recall   
    for target_recall in target_recall_list:
        print("target_recall:", target_recall)
        print("sel_list:", target_sel_list)
        data = {}
        plt.figure(figsize=(10, 6))
        for idx, algo in enumerate(range_query_algo):
            _algo = algo.replace("_", "-")
            print(_algo, " & ", end="")
            for dataset in dataset_list:
                param_list = get_param_by_recall(algo, dataset, target_sel_list, target_recall)
                line_style = line_styles[idx % len(line_styles)]
                marker = markers[idx % len(markers)]
                # print("algo:", algo, " qps_values:", qps_values)
                
                max_val = param_list[0]
                min_val = param_list[0]
                for v in param_list:
                    if max_val < v:
                        max_val = v
                    if min_val == -1 or (min_val > v and v != -1):
                        min_val = v
                

                data[algo] = param_list
                for i in range(len(param_list)):
                    if param_list[i] == -1:
                        print("\cellcolor{blue!10}-" , end="")
                    else:
                        val = int(param_list[i])
                        if max_val == min_val:
                            weight = 10
                        else:
                            weight = int((val-min_val) / (max_val-min_val) * 90 + 10)
                        color = f"\cellcolor{{blue!{weight}}}"
                        textcolor = str(int(val))
                        if weight > 60:
                            textcolor = f"\\textcolor{{white}}{{{str(int(val))}}}"
                        val = color + textcolor
                        print(val, end="")
                    if i != len(param_list)-1 or dataset != dataset_list[-1]:
                        print(" & ", end="")
                    else:
                        print("\\\\")

