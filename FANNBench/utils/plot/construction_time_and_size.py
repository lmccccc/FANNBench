import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import pandas as pd
import sys


# title = "Paper Dataset dim N query_size label_method query_method distribution label_range query_label_cnt K Threads M serf_M nprobe ef_construction ef_search gamma M_beta alpha L  partition_size_M beamSize split_factor shift_factor final_beam_multiply kgraph_L iter S R B kgraph_M weight_search Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery File Memory"
# title_list = title.split(' ')

algorithms = ['Acorn', 'DiskANN', 'HNSW', "IVFPQ", "Milvus_IVFPQ", "Milvus_HNSW", "NHQ_kgraph", "NHQ_nsw", 
              "RII", "SeRF", "VamanaTree", "iRangeGraph", "WST_opt"]
indexsize_res_map = {}
constime_res_map = {}
memory_map = {}
# main function
if __name__ == "__main__":
    file_path = sys.argv[1]
    dataset = sys.argv[2]
    label_method = sys.argv[3]

    
    with open(file_path, 'r') as file:
        data = pd.read_csv(file)
    # print(data)
    # get index size and construction time
    # due to various configurations, we need to find the suitable row. use the latest row as the reference
    for index, row in data.iterrows():
        index_size = row["IndexSize"]
        cons_time = row["ConstructionTime"]
        memory = row["Memory"]
        algo = row["Paper"]
        if isinstance(index_size, (int, float)) and index_size > 0:
            if row["Dataset"] == dataset: # dataset match
                indexsize_res_map[algo] = index_size
        # print("index size:", row["IndexSize"])
            
        if isinstance(cons_time, (int, float)) and cons_time > 0: 
            if row["Dataset"] == dataset: # dataset match
                constime_res_map[algo] = cons_time

        if isinstance(cons_time, (int, float)) and memory > 0:
            if row["Dataset"] == dataset: # dataset match
                memory_map[algo] = memory

    # plot size
    algorithms = list(indexsize_res_map.keys())
    index_sizes = [indexsize_res_map[algo] for algo in algorithms]


    fig, ax = plt.subplots()
    ax.bar(algorithms, index_sizes)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Index Size (MB)')
    ax.set_title('Index Size by Algorithm on ' + dataset)
    plt.xticks(rotation=45, ha='right')
    ax.set_yscale('log', base=10)
    plt.tight_layout()
    plt.show()
    size_file = "plot/index_size_plot_" + label_method + '_' + dataset + ".png"
    plt.savefig(size_file)  # Save the plot as an image file

    # plot cons time
    algorithms = list(constime_res_map.keys())
    construction_times = [constime_res_map[algo] for algo in algorithms]
    fig, ax = plt.subplots()
    ax.bar(algorithms, construction_times)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Construction Time (s)')
    ax.set_title('Construction Time by Algorithm on ' + dataset)
    plt.xticks(rotation=45, ha='right')
    ax.set_yscale('log', base=10)
    plt.tight_layout()
    plt.show()
    time_file = "plot/construction_time_plot_" + label_method + '_' + dataset + ".png"
    plt.savefig(time_file)  # Save the plot as an image file

    
    # plot memory
    algorithms = list(memory_map.keys())
    mem = [memory_map[algo] for algo in algorithms]
    fig, ax = plt.subplots()
    ax.bar(algorithms, mem)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Memory Footprint (MB)')
    ax.set_title('Construction Memory Footprint by Algorithm on ' + dataset)
    plt.xticks(rotation=45, ha='right')
    ax.set_yscale('log', base=10)
    plt.tight_layout()
    plt.show()
    memory_file = "plot/memory_plot_" + label_method + '_' + dataset + ".png"
    plt.savefig(memory_file)  # Save the plot as an image file

    