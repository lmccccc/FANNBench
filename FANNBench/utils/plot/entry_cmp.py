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



all_algo = [
          "HNSW",
          "HNSW_bottom",
          "ACORN",
          "ACORN_bottom",
          "UNIFY_hybrid",
          "UNIFY_bottom",
          "UNIFY_middle",
          "UNIFY_left",
          "UNIFY_right",
          "DSG",
          "DSG_left",
          "DSG_right",
          "SeRF",
          "SeRF_left",
          "SeRF_right",
          ]

serf_algo = [
    "SeRF",
    "SeRF_left",
    "SeRF_right"
]
dsg_algo = [
    "DSG",
    "DSG_left",
    "DSG_right"
]
unify_algo = [
    "UNIFY_hybrid",
    "UNIFY_middle",
    "UNIFY_left",
    "UNIFY_right"
]

algo_group = [
    unify_algo,
    serf_algo,
    dsg_algo
]

line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
query_map = {}


def get_qps(algo, recall):
    best_qps = 1
    best_cpq = 2**31-1
    for item in algo.values():
        if item["Recall"] >= recall-0.005:
            if item["QPS"] > best_qps:
                best_qps = item["QPS"]
            if item["CompsPerQuery"] < best_cpq and item["CompsPerQuery"] > 1:
                best_cpq = item["CompsPerQuery"]
    
    return best_qps, best_cpq

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
    
    dataset="sift10M"
    distribution = "random"
    label_range = 100000

    target_recall = 0.9
    target_sel_list = [0.01, 0.1, 0.5, 1]
    target_id_list = [sel2id[sel] for sel in target_sel_list]
    target_qrange_list = [sel2range[sel] for sel in target_sel_list]

    target_qrange = [sel2range[target_sel] for target_sel in target_sel_list]
    target_id = [sel2id[target_sel] for target_sel in target_sel_list]


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
        if algo not in all_algo or row["label_range"] != label_range:
            continue
        if row["query_label_cnt"] not in target_qrange and row["query_label_cnt"] not in target_id:
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
        
        if recall < target_recall-0.1:
            continue
        
        res_turple = {"Recall": recall, "QPS": qps, "CompsPerQuery": comps}
        if "SeRF" in algo or "DSG" in algo:
            efs = row["ef_search"]
            query_map[algo][sel][efs] = res_turple
        elif "UNIFY_" in algo:
            efs = row["ef_search"]
            al = row["AL"]
            query_map[algo][sel][efs * 101 + al * 698] = res_turple
        
    # print("UNIFY_hybrid:", query_map["UNIFY_hybrid"])
    print("Algorithm & 0.01 & 0.1 & 0.5 & 1\\\\")
    algo_res = {}
    for algos in algo_group:
        for algo in algos:
            algo_res[algo] = {}

            for target_sel in target_sel_list:
                if algo not in query_map.keys():
                    print(algo, "not in query_map")
                if target_sel not in query_map[algo].keys():
                    print(target_sel, "not in ", algo)
                    print(query_map[algo].keys())
                target_algo = query_map[algo][target_sel]
                qps, cpq = get_qps(target_algo, target_recall)
                algo_res[algo][target_sel] = {}
                algo_res[algo][target_sel]["QPS"] = qps
                algo_res[algo][target_sel]["CPQ"] = cpq

                if "UNIFY_" in algo and algo != "UNIFY_hybrid":
                    qps_div = (qps-algo_res["UNIFY_hybrid"][target_sel]["QPS"]) / algo_res["UNIFY_hybrid"][target_sel]["QPS"]
                    cpq_div = (cpq-algo_res["UNIFY_hybrid"][target_sel]["CPQ"]) / algo_res["UNIFY_hybrid"][target_sel]["CPQ"]
                    algo_res[algo][target_sel]["QPS_div"] = int(qps_div * 100)
                    algo_res[algo][target_sel]["CPQ_div"] = int(cpq_div * 100)
                elif "SeRF_" in algo:
                    qps_div = (qps-algo_res["SeRF"][target_sel]["QPS"]) / algo_res["SeRF"][target_sel]["QPS"]
                    cpq_div = (cpq-algo_res["SeRF"][target_sel]["CPQ"]) / algo_res["SeRF"][target_sel]["CPQ"]
                    algo_res[algo][target_sel]["QPS_div"] = int(qps_div * 100)
                    algo_res[algo][target_sel]["CPQ_div"] = int(cpq_div * 100)
                elif "DSG_" in algo:
                    qps_div = (qps-algo_res["DSG"][target_sel]["QPS"]) / algo_res["DSG"][target_sel]["QPS"]
                    cpq_div = (cpq-algo_res["DSG"][target_sel]["CPQ"]) / algo_res["DSG"][target_sel]["CPQ"]
                    algo_res[algo][target_sel]["QPS_div"] = int(qps_div * 100)
                    algo_res[algo][target_sel]["CPQ_div"] = int(cpq_div * 100)

    print("dataset: ", dataset)
    for algos in algo_group:
        for algo in algos:
            print(algo + " & ", end="")
            for idx, target_sel in enumerate(target_sel_list):
                if algo == "SeRF" or algo == "DSG" or algo == "UNIFY_hybrid":
                    print(f"%{algo_res[algo][target_sel]['QPS']:.2f}$", end="")
                else:
                    print(f"{algo_res[algo][target_sel]['QPS_div']:.2f}\%", end="")
                if idx < len(target_sel_list) - 1:
                    print(" & ", end="")
                else:
                    print("\\\\")

                # qps_head_div = (qps_head-qps)/qps
                # cpq_head_div = (cpq_head - cpq)/cpq
                # qps_tail_div = (qps_tail-qps)/qps
                # cpq_tail_div = (cpq_tail - cpq)/cpq

            


            # serf = query_map["SeRF"][target_sel]
            # qps, cpq = get_qps(serf, target_recall)
            # SeRF_left = query_map["SeRF_left"][target_sel]
            # qps_head, cpq_head = get_qps(SeRF_left, target_recall)
            # qps_head_div = (qps_head-qps)/qps
            # cpq_head_div = (cpq_head - cpq)/cpq
            # SeRF_right = query_map["SeRF_right"][target_sel]
            # qps_tail, cpq_tail = get_qps(SeRF_right, target_recall)
            # qps_tail_div = (qps_tail-qps)/qps
            # cpq_tail_div = (cpq_tail - cpq)/cpq
            # # print("SeRF & -  & %.2f(baseline) & %.2f(%.2f) & %.2f(%.2f) \\\\" % (qps, qps_head, qps_head_div, qps_tail, qps_tail_div))
            # print(f"SeRF & -  & default & {int(qps_head_div * 100)}\% & {int(qps_tail_div * 100)}\% \\\\")
            # # print("SeRF & -  & %.2f(baseline) & %.2f(%.2f) & %.2f(%.2f) & - & - \\\\" % (cpq, cpq_head, cpq_head_div, cpq_tail, cpq_tail_div))

            # dsg = query_map["DSG"][target_sel]
            # qps, cpq = get_qps(dsg, target_recall)
            # DST_left = query_map["DSG_left"][target_sel]
            # qps_head, cpq_head = get_qps(DST_left, target_recall)
            # qps_head_div = (qps_head-qps)/qps
            # cpq_head_div = (cpq_head - cpq)/cpq
            # DST_right = query_map["DSG_right"][target_sel]
            # qps_tail, cpq_tail = get_qps(DST_right, target_recall)
            # qps_tail_div = (qps_tail-qps)/qps
            # cpq_tail_div = (cpq_tail - cpq)/cpq
            # # print("DSG & -  & %.2f(baseline) & %.2f(%.2f) & %.2f(%.2f) \\\\" % (qps, qps_head, qps_head_div, qps_tail, qps_tail_div))
            # print(f"DSG & - & default & {int(qps_head_div * 100)}\% & {int(qps_tail_div * 100)}\% \\\\")
            # # print("DSG & -  & %.2f(baseline) & %.2f(%.2f) & %.2f(%.2f) & - & - \\\\" % (cpq, cpq_head, cpq_head_div, cpq_tail, cpq_tail_div))



    
    
    