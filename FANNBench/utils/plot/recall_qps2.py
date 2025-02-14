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
              "RII", "SeRF", "iRangeGraph", "UNIFY", "DSG"]
range_query_algo = ["ACORN", "HNSW", "IVFPQ", "Milvus_IVFPQ", "Milvus_HNSW", "WST_opt", "WST_vamana", 
              "SeRF", "iRangeGraph", "UNIFY", "DSG"]
cpq_range_query_algo = ["ACORN", "HNSW", "WST_opt", "WST_vamana", 
              "SeRF", "iRangeGraph", "UNIFY", "DSG"]
comps_algo = ["ACORN", "DiskANN", "HNSW", "NHQ_kgraph", "NHQ_nsw", "WST_opt", "WST_vamana", 
              "SeRF", "iRangeGraph","UNIFY", "DSG"]
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '|', '+', 'x', 'X', 'H', '_']
query_map = {}


def get_qps_by_recall(algo, query_label_list, target_recall):
    qps_values = []
    if algo not in query_map.keys():
        qps_values.extend([0] * len(query_label_list))
        return qps_values
    for query_label_cnt in query_label_list:
        if query_label_cnt not in query_map[algo].keys():
            qps_values.append(0)
            continue
        max_qps = -1
        for entry in query_map[algo][query_label_cnt]:
            if entry["Recall"] >= target_recall-0.005 and entry["QPS"] > max_qps:
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
        min_comps = int(1e9)
        for entry in query_map[algo][query_label_cnt]:
            if entry["Recall"] >= target_recall-0.005 and entry["CompsPerQuery"] < min_comps:
                min_comps = entry["CompsPerQuery"]
        if min_comps == int(1e9):
            min_comps = 0
        comp_values.append(min_comps)
    return comp_values

def savedata(data, _file, xlist):
    # data = pd.DataFrame(data.values())
    # print("oiri data:")
    # print(data)
    data['Selectivity'] = xlist
    data = pd.DataFrame(data)
    # print("dataframe:")
    # print(data)
    # save 
    data.to_csv(_file, index=False)
    # data.to_csv(_file,index=False,sep=' ')
    # Open the file in write mode
    # with open(_file, 'w') as file:
    #     # write the data
    #     for line in data:
    #         # write list
    #         for idx, info in enumerate(line):
    #             file.write(str(info))
    #             if idx < len(line) - 1:
    #                 file.write(' ')
                
    #         file.write('\n')
    #     file.close()


    
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
    for index, row in data.iterrows():
        if row["Dataset"] != dataset or \
            distribution != row["distribution"] or \
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
        algo = row["Paper"]
        if recall <= 0:
            continue
        if recall > 1:
            recall = recall / 100
        if algo not in query_map.keys():
            query_map[algo] = {}
        if row["query_label_cnt"] not in query_map[algo].keys():
            query_map[algo][row["query_label_cnt"]] = []

        res_turple = {"Recall": recall, "QPS": qps, "CompsPerQuery": comps}
        query_map[algo][query_label_cnt].append(res_turple)
    

    query_label_list = [1000 * i for i in range(1, 11)]
    sel_list = [i/100 for i in range(1, 11)]
    # plot by selectivity
    # plot 0.9 recall
    target_recall_list = [0.8, 0.9, 0.95, 0.99]

    for target_recall in target_recall_list:
        data = {}
        plt.figure(figsize=(10, 6))
        for idx, algo in enumerate(range_query_algo):
            qps_values = get_qps_by_recall(algo, query_label_list, target_recall)
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            # print("algo:", algo, " qps_values:", qps_values)
            plt.plot(sel_list, qps_values, label=algo, linestyle=line_style, marker=marker)
            data[algo] = qps_values

        plt.xlabel('Selectivity')
        plt.ylabel('QPS')
        plt.yscale('log')
        plt.title('QPS at {recall} Recall with {query_method} Label'.format(recall=target_recall, query_method=str(label_range)+ ' ' + distribution))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        label = "qps_" + str(target_recall) + "recall_" + str(label_range) + "label_" + distribution + "_" + dataset
        file = "plot/" + label + ".png"
        plt.savefig(file)
        # print("save file to ", file)
        # Open the file in write mode
        xlsfile = xlspath + label + tail
        # writ to csv file
        print("algo:", range_query_algo)
        savedata(data, xlsfile, sel_list)
    
    
    for target_recall in target_recall_list:
        plt.figure(figsize=(10, 6))
        data = {}
        for idx, algo in enumerate(cpq_range_query_algo):
            comps_values = get_comps_by_recall(algo, query_label_list, target_recall)
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            data[algo] = comps_values
            # print("algo:", algo, " comps_values:", qps_values)
            plt.plot(sel_list, comps_values, label=algo, linestyle=line_style, marker=marker)

        plt.xlabel('Selectivity')
        plt.ylabel('Comparasions per Query')
        plt.yscale('log')
        plt.title('CPQ at {recall} Recall with {query_method} Label'.format(recall=target_recall, query_method=str(label_range)+ ' ' + distribution))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        label = "cpq_" + str(target_recall) + "recall_" + str(label_range) + "label_" + distribution + "_" + dataset
        file = "plot/" + label + ".png"
        plt.savefig(file)
        print("save file to ", file)
        # Open the file in write mode
        # writ to csv file
        print("algo:", cpq_range_query_algo)
        xlsfile = xlspath + label + tail
        savedata(data, xlsfile, sel_list)


    small_query_label_list = [100 * i for i in range(1, 11)]
    sel_list = [1/1000 * i for i in range(1, 11)]
    for target_recall in target_recall_list:
        data = {}
        plt.figure(figsize=(10, 6))
        for idx, algo in enumerate(range_query_algo):
            qps_values = get_qps_by_recall(algo, small_query_label_list, target_recall)
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            data[algo] = qps_values
            # print("algo:", algo, " qps_values:", qps_values)
            plt.plot(sel_list, qps_values, label=algo, linestyle=line_style, marker=marker)

        plt.xlabel('Selectivity')
        plt.ylabel('QPS')
        plt.yscale('log')
        plt.title('QPS at {recall} Recall with {query_method} Label'.format(recall=target_recall, query_method=str(label_range)+ ' ' + distribution))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        label = "smallsel_qps_" + str(target_recall) + "recall_" + str(label_range) + "label_" + distribution + "_" + dataset
        file = "plot/" + label + ".png"
        plt.savefig(file)
        print("save file to ", file)
        # Open the file in write mode
        print("algo:", range_query_algo)
        xlsfile = xlspath + label + tail
        savedata(data, xlsfile, sel_list)
    
    
    for target_recall in target_recall_list:
        data = {}
        plt.figure(figsize=(10, 6))
        for idx, algo in enumerate(cpq_range_query_algo):
            comps_values = get_comps_by_recall(algo, small_query_label_list, target_recall)
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            data[algo] = comps_values
            # print("algo:", algo, " comps_values:", qps_values)
            plt.plot(sel_list, comps_values, label=algo, linestyle=line_style, marker=marker)

        plt.xlabel('Selectivity')
        plt.ylabel('Comparasions per Query')
        plt.yscale('log')
        plt.title('CPQ at {recall} Recall with {query_method} Label'.format(recall=target_recall, query_method=str(label_range)+ ' ' + distribution))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        label = "smallsel_cpq_" + str(target_recall) + "recall_" + str(label_range) + "label_" + distribution + "_" + dataset
        file = "plot/" + label + ".png"
        plt.savefig(file)
        # Open the file in write mode
        # writ to csv file
        print("algo:", cpq_range_query_algo)
        xlsfile = xlspath + label + tail
        savedata(data, xlsfile, sel_list)

    large_query_label_list = [10000 * i for i in range(1, 11)]
    sel_list = [1/10 * i for i in range(1, 11)]
    for target_recall in target_recall_list:
        data = {}
        plt.figure(figsize=(10, 6))
        for idx, algo in enumerate(range_query_algo):
            qps_values = get_qps_by_recall(algo, large_query_label_list, target_recall)
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            data[algo] = qps_values
            # print("algo:", algo, " qps_values:", qps_values)
            plt.plot(sel_list, qps_values, label=algo, linestyle=line_style, marker=marker)

        plt.xlabel('Selectivity')
        plt.ylabel('QPS')
        plt.yscale('log')
        plt.title('QPS at {recall} Recall with {query_method} Label'.format(recall=target_recall, query_method=str(label_range)+ ' ' + distribution))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        label = "largesel_qps_" + str(target_recall) + "recall_" + str(label_range) + "label_" + distribution + "_" + dataset
        file = "plot/" + label + ".png"
        plt.savefig(file)
        print("save file to ", file)
        # Open the file in write mode
        print("algo:", range_query_algo)
        xlsfile = xlspath + label + tail
        savedata(data, xlsfile, sel_list)
    
    
    for target_recall in target_recall_list:
        data = {}
        plt.figure(figsize=(10, 6))
        for idx, algo in enumerate(cpq_range_query_algo):
            comps_values = get_comps_by_recall(algo, large_query_label_list, target_recall)
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            data[algo] = comps_values
            # print("algo:", algo, " comps_values:", qps_values)
            plt.plot(sel_list, comps_values, label=algo, linestyle=line_style, marker=marker)

        plt.xlabel('Selectivity')
        plt.ylabel('Comparasions per Query')
        plt.yscale('log')
        plt.title('CPQ at {recall} Recall with {query_method} Label'.format(recall=target_recall, query_method=str(label_range)+ ' ' + distribution))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        label = "largesel_cpq_" + str(target_recall) + "recall_" + str(label_range) + "label_" + distribution + "_" + dataset
        file = "plot/" + label + ".png"
        plt.savefig(file)
        # Open the file in write mode
        # writ to csv file
        print("algo:", cpq_range_query_algo)
        xlsfile = xlspath + label + tail
        savedata(data, xlsfile, sel_list)