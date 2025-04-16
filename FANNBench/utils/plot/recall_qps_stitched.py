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



range_algo="DiskANN_Stitched"

line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
query_map = {}


def get_comps_by_recall(stitched_R, sel):
    if stitched_R not in query_map.keys():
        return [], [], []
    if sel not in query_map[stitched_R].keys():
        return [], [], []
    tmp = []
    for item in query_map[stitched_R][sel].keys():
        tmp.append(query_map[stitched_R][sel][item])
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
    
    # dataset="YTRGB1m"
    # label_range = 100000
    target_id_list = [sel2id[sel] for sel in target_sel_list]
    target_qrange_list = [sel2range[sel] for sel in target_sel_list]

    for index, row in data.iterrows():
        if row["Dataset"] != dataset or \
            label_cnt != row["label_cnt"] or \
            row["Threads"] != 1:
            continue
        query_label = row["query_label"]
        recall = row["Recall"]
        qps = row["QPS"]
        comps = row["CompsPerQuery"]
        algo = row["Paper"]
        if algo != range_algo:
            continue
        if row["label_range"] != label_range:
            continue
        if recall <= 0:
            continue
        if recall > 1:
            recall = recall / 100
        if row["query_label"] in target_id_list:
            sel = id2sel[row["query_label"]]
        else:
            continue
        stitched_R = int(row["Stitched_R"])
        if stitched_R not in query_map.keys():
            query_map[stitched_R] = {}

        if sel not in query_map[stitched_R].keys():
            query_map[stitched_R][sel] = {}
        
        if recall < 0.3:
            continue
        res_turple = {"Recall": recall, "QPS": qps, "CompsPerQuery": comps}
        
        if  algo == "DiskANN_Stitched":
            l = row["L"]
            query_map[stitched_R][sel][l] = res_turple
    
    columns = ['stitched_R', 'QPS', 'CPQ']
    # print(query_map)
    for sel in target_sel_list:
        df = pd.DataFrame(columns=columns)
        plt.figure(figsize=(10, 6))
        for idx, stitched_R in enumerate(query_map.keys()):
            recall, qps, cpq = get_comps_by_recall(stitched_R, sel)
            for i in range(len(qps)):
                row = {"stitched_R": f"stitched_R={stitched_R}", "QPS": qps[i], "CPQ": cpq[i], "recall": recall[i]}
                new_row = pd.DataFrame([row])
                df = pd.concat([df, new_row], ignore_index=True)
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            # print("algo ", algo, "recall: ", recall, "qps: ", qps)
            plt.plot(recall, qps, label=f"M={stitched_R}", linestyle=line_style, marker=marker)

        plt.xlabel('recall')
        plt.ylabel('QPS')
        plt.yscale('log')
        # plt.xscale('log')
        plt.title('SeRF recall/qps with {query_method} Label'.format(query_method=str(label_range)+ ' ' + distribution))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        label = "recall_qps_stitched_" + str(label_range) + "label_" + str(sel) + "sel_" + distribution + "_" + dataset
        file = "plot/png/" + label + ".png"
        plt.savefig(file)
        print("save image to ", file)


        label = "recall_qps_stitched_" + str(label_range) + "label_" + str(sel) + "sel_" + distribution + "_" + dataset
        xlsfile = xlspath + label + tail
        savedata(df, xlsfile)
