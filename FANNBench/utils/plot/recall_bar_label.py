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

# algorithms = ["DiskANN", "diskann_stitched", "HNSW", "IVFPQ", "NHQ_kgraph", "NHQ_nsw", "Milvus_IVFPQ", "Milvus_HNSW", "WST_opt", "WST_vamana", 
            #   "RII", "SeRF", "iRangeGraph", "UNIFY", "UNIFY_hybrid", "DSG"]
range_query_algo = ["Milvus_IVFPQ", 
          "Milvus_HNSW", 
          "IVFPQ",
          "HNSW",
          "ACORN", 
          "DiskANN",
          "DiskANN_Stitched",
          "NHQ_kgraph",
          "NHQ_nsw"]
cpq_range_query_algo = [
          "HNSW",
          "ACORN", 
          "DiskANN",
          "DiskANN_Stitched",
          "NHQ_kgraph",
          "NHQ_nsw"]
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

        if len(query_map[algo][sel]) > 20:
            tmp_len = len(query_map[algo][sel])
            tmp = query_map[algo][sel][tmp_len-20:]
        else:
            tmp = query_map[algo][sel]
        # sort by recall
        tmp = sorted(tmp, key=lambda x: x["Recall"])
        result = []

        start = 0
        for i in range(len(tmp)):
            if tmp[i]["QPS"] > 1:
                result.append(tmp[i])
                start = i+1
                break

        # if QPS is small than previous one, remove it
        # print(algo, sel)
        # print(tmp)
        # for i in range(start, len(tmp)):
        #     # 获取当前元素和前一个元素
        #     current = tmp[i]
        #     prev = result[-1]
        #     # 检查 a 是否单调递增且 b 是否单调递减
        #     if current["QPS"] <= prev["QPS"]:
        #         # 如果满足条件，将当前元素加入结果列表
        #         if current["Recall"] > prev["Recall"]:
        #             result.append(current)
        #         else:
        #             # print(algo, sel, " replace ", result[-1], " with ", current)
        #             result[-1] = current
        # if len(result) < 3:
        #     for r in result:
        #         if (result[-1]["Recall"] > 0.95 * target_recall):
        #             linear_result = r["QPS"]
        #             break
        #         else:
        #             linear_result = 1
        #     print("invalid result for ", algo, sel, " result size:", len(result), " save qps:", linear_result)
        # else:
        #     # interpolation
        #     if (result[0]["Recall"] > 0.95 * target_recall):
        #             linear_result = result[0]["QPS"]
        #     else:
        #         x = [val["Recall"] for val in result]
        #         y = [val["QPS"] for val in result]
        #         linear_interp = interp1d(x, y, kind='quadratic', fill_value='extrapolate')
        #         linear_result = linear_interp(target_recall)
        max_qps = 1
        for entry in query_map[algo][sel]:
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
        if len(query_map[algo][sel]) > 10:
            tmp_len = len(query_map[algo][sel])
            tmp = query_map[algo][sel][tmp_len-10:]
        else:
            tmp = query_map[algo][sel]
        # sort by recall
        tmp = sorted(tmp, key=lambda x: x["Recall"])
        result = []

        start = 0
        for i in range(len(tmp)):
            if tmp[i]["QPS"] > 1:
                result.append(tmp[i])
                start = i+1
                break

        # if QPS is small than previous one, remove it
        # for i in range(start, len(tmp)):
        #     # 获取当前元素和前一个元素
        #     current = tmp[i]
        #     prev = result[-1]
        #     # 检查 a 是否单调递增且 b 是否单调递减
        #     if current["CompsPerQuery"] >= prev["CompsPerQuery"]:
        #         # 如果满足条件，将当前元素加入结果列表
        #         if current["Recall"] > prev["Recall"]:
        #             result.append(current)
        #         else:
        #             # print(algo, sel, " replace ", result[-1], " with ", current)
        #             result[-1] = current

        # if len(result) < 3:
        #     for r in result:
        #         if (result[-1]["Recall"] > 0.95 * target_recall):
        #             linear_result = r["CompsPerQuery"]
        #             break
        #         else:
        #             linear_result = 1
        #     print("invalid result for ", algo, sel, " result size:", len(result), " save cmp:", linear_result)
        # else:
        #     # interpolation
        #     if (result[0]["Recall"] > 0.95 * target_recall):
        #             linear_result = result[0]["CompsPerQuery"]
        #     else:
        #         x = [val["Recall"] for val in result]
        #         y = [val["CompsPerQuery"] for val in result]
        #         linear_interp = interp1d(x, y, kind='quadratic', fill_value='extrapolate')
        #         linear_result = linear_interp(target_recall)

        for entry in query_map[algo][sel]:
            if entry["Recall"] >= target_recall-0.01 and entry["CompsPerQuery"] < min_comps:
                min_comps = entry["CompsPerQuery"]
        if min_comps == int(1e9):
            min_comps = 1
        # assert(min_comps > 0)
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
    
    # dataset="spacev10m"
    distribution = "random"
    label_range = 500
    target_recall_list = [0.9, 0.95]
    target_id_list = [sel2id[sel] for sel in target_sel_list]

    for index, row in data.iterrows():
        if row["Dataset"] != dataset or \
            distribution != row["distribution"] or \
            label_range != row["label_range"] or \
            label_cnt != row["label_cnt"] or \
            row["Threads"] != 1:
            continue
        if query_label_cnt == 1 and label_cnt > 1 and row["query_label"] != query_label:
            continue
        query_label = row["query_label"]
        recall = row["Recall"]
        qps = row["QPS"]
        comps = row["CompsPerQuery"]
        algo = row["Paper"]
        if row["query_label"] not in target_id_list:
            continue
        if recall <= 0:
            continue
        if recall > 1:
            recall = recall / 100
        if algo not in query_map.keys():
            query_map[algo] = {}
        if row["query_label"] in target_id_list:
            sel = id2sel[row["query_label"]]
        if sel not in query_map[algo].keys():
            query_map[algo][sel] = []

        res_turple = {"Recall": recall, "QPS": qps, "CompsPerQuery": comps}
        query_map[algo][sel].append(res_turple)
    

    # print("query_map:", query_map)
    # plot by selectivity
    # plot 0.9 recall   
    for target_recall in target_recall_list:
        data = {}
        # plt.figure(figsize=(10, 6))
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
            # plt.bar(x, y)
            # plt.xlabel('Algorithms')
            # plt.ylabel('QPS')
            # plt.yscale('log')
            # plt.title('QPS at {recall} Recall, {sel} Selectivity, and {query_method} Label'.format(recall=target_recall, sel=sel, query_method=str(label_range)+ ' ' + distribution))
            # # plt.legend()
            # plt.grid(True)
            # # plt.tight_layout()
            # plt.show()
            # label = "bar_label_qps_" + str(target_recall) + "recall_" + str(sel) + "sel_" + str(label_range) + "label_" + distribution + "_" + dataset
            # file = "plot/" + label + ".png"
            # plt.savefig(file)
            # print("save file to ", file)
            # Open the file in write mode
            # writ to csv file
        label = "bar_label_qps_" + str(target_recall) + "recall_" + str(sel) + "label_" + distribution + "_" + dataset
        xlsfile = xlspath + label + tail
        # print("algo:", range_query_algo)
        savedata(data, xlsfile, target_sel_list)
    
    
    for target_recall in target_recall_list:
        # plt.figure(figsize=(10, 6))
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
            # plt.bar(x, y)

            # plt.xlabel('Selectivity')
            # plt.ylabel('Comparasions per Query')
            # plt.yscale('log')
            # plt.title('bar_CPQ at {recall} Recall with {query_method} Label'.format(recall=target_recall, query_method=str(label_range)+ ' ' + distribution))
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()
            # label = "bar_label_cpq_" + str(target_recall) + "recall_" + str(sel) + "sel_" + str(label_range) + "label_" + distribution + "_" + dataset
            # file = "plot/" + label + ".png"
            # plt.savefig(file)
            # print("save file to ", file)
            # Open the file in write mode
            # writ to csv file
        label = "bar_label_cpq_" + str(target_recall) + "recall_" + distribution + "_" + dataset
        xlsfile = xlspath + label + tail
        savedata(data, xlsfile, target_sel_list)
