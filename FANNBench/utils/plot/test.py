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
          "UNIFY",
          "UNIFY_hybrid",]

line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
query_map = {}


def get_comps_by_recall(algo, sel, recall):
    if algo not in query_map.keys():
        return 1,1
    if sel not in query_map[algo].keys():
        return 1,1
    
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
    target_recall = 0.99
    target_id_list = [sel2id[sel] for sel in target_sel_list]
    target_qrange_list = [sel2range[sel] for sel in target_sel_list]

    # times = []
    # for index, row in data.iterrows():
    #     if row["Dataset"] == "sift10M" and \
    #         row["distribution"] == "random" and \
    #         row["Paper"] == "ACORN" and \
    #         row["Threads"] == 128 and \
    #         row["ef_construction"] == 1000 and \
    #         row["label_range"] == 100000 and \
    #         row["ConstructionTime"] > 0:
    #         print(row["Dataset"], row["Paper"], row["ConstructionTime"], row["IndexSize"], row["Memory"])
    #         times.append(row["ConstructionTime"])
    
    # times = np.array(times)
    # avg = np.mean(times)
    # std = np.std(times)
    # var = np.var(times)
    # sub = np.abs(times - avg)
    # extreme = np.max(sub)
    # pct = extreme / avg
    # std2 = std / avg
    # print("avg: ", avg, " std: ", std, " var: ", var, " extreme: ", extreme, " pct: ", pct, " std2: ", std2)
    


        # if row["Dataset"] == "sift10M" and \
        #     row["distribution"] == "random" and \
        #     row["Paper"] == "ACORN" and \
        #     row["Threads"] == 1 and \
        #     row["ef_construction"] == 1000 and \
        #     row["ef_search"] == 100 and \
        #     row["label_range"] == 100000 and \
        #     (row["query_label_cnt"] == 50000 or row["query_label_cnt"] == 6):
        #     print(row["Dataset"], row["Paper"], row["Recall"], row["QPS"], row["CompsPerQuery"])
    qps = []
    for index, row in data.iterrows():
        if row["Dataset"] == "sift10M" and \
            row["distribution"] == "random" and \
            row["Paper"] == "UNIFY" and \
            row["Threads"] == 1 and \
            row["ef_construction"] == 1000 and \
            row["ef_search"] == 150 and \
            row["label_range"] == 100000 and \
            (row["query_label_cnt"] == 50000 or row["query_label_cnt"] == 6):
            qps.append(row["QPS"])
            print(row["Dataset"], row["Paper"], row["Recall"], row["QPS"], row["CompsPerQuery"])

    qps = np.array(qps)
    avg = np.mean(qps)
    std = np.std(qps)
    var = np.var(qps)
    sub = np.abs(qps - avg)
    extreme = np.max(sub)
    pct = extreme / avg
    std2 = std / avg
    print("avg: ", avg, " std: ", std, " var: ", var, " extreme: ", extreme, " pct: ", pct, " std2: ", std2)

