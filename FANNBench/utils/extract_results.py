import numpy as np
import sys
import os
import pandas as pd

# title = "Paper	Dataset	label method	Threads	dis method	Data size/Query size/dim	K	M(hnsw) R(diskann)		efc(hnsw) beamsize(rfann)	efs  L(diskann)	gamma(acorn) multiply(rfann)	Mbeta	alpha	nprobe	beamSize	final beam multiply	split factor	shift factor	final multiplies	partition size M	recall	QPS	selectivity	ConstructionTime	IndexSize	CompsPerQuery	File"
title = "Paper Dataset dim N query_size label_method query_method distribution label_range query_label_cnt K Threads M serf_M nprobe ef_construction ef_search gamma M_beta alpha L  partition_size_M beamSize split_factor shift_factor final_beam_multiply kgraph_L iter S R B kgraph_M weight_search Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery File"
title_list = title.split(' ')
#algo_list = ["RangeFilteredANN"]
def get_info_from_wst(df, lines):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if('super-postfiltering' in line):
            # ('super-postfiltering_2_0.5_1_10_16', 0.6053700000000002, 0.11043572425842285, 380.4889991283417, 2, 7.257514953613281)
            # remove (), split by ','
            info = line.split('(')[1].split(')')[0].split(',')
            recall = float(info[1])
            cons_time = float(info[3])
            df.at[0, "Recall"] = recall
            df.at[0, "ConstructionTime"] = cons_time

        elif "qps" in line:
            df.at[0, "QPS"] = float(line.split(":")[1])
        elif "avg comp" in line:
            df.at[0, "CompsPerQuery"] = float(line.split(":")[1])
    
    # assert if any of the value is empty
    if(df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1 or df.at[0, "CompsPerQuery"] == -1):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_vtree(df, lines):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if('vamana-tree' in line):
            # ('super-postfiltering_2_0.5_1_10_16', 0.6053700000000002, 0.11043572425842285, 380.4889991283417, 2, 7.257514953613281)
            # remove (), split by ','
            info = line.split('(')[1].split(')')[0].split(',')
            recall = float(info[2])
            cons_time = float(info[4])
            df.at[0, "Recall"] = recall
            df.at[0, "ConstructionTime"] = cons_time

        elif "qps" in line:
            df.at[0, "QPS"] = float(line.split(":")[1])
        elif "avg comp" in line:
            df.at[0, "CompsPerQuery"] = float(line.split(":")[1])
    
    # assert if any of the value is empty
    if(df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1 or df.at[0, "CompsPerQuery"] == -1):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_ivfpq(df, lines):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if('index construction total cost' in line):
            info = line.split(':')
            cons_time = float(info[1])
            df.at[0, "ConstructionTime"] = cons_time
        elif('recall@' in line):
            info = line.split(':')
            recall = float(info[1])
            df.at[0, "Recall"] = recall
        elif "qps" in line:
            df.at[0, "QPS"] = float(line.split(":")[1])
    
    # assert if any of the value is empty
    if(df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_hnsw(df, lines):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if('index construction total cost' in line):
            info = line.split(':')
            cons_time = float(info[1])
            df.at[0, "ConstructionTime"] = cons_time
        elif('recall@' in line):
            info = line.split(':')
            recall = float(info[1])
            df.at[0, "Recall"] = recall
        elif "qps" in line:
            df.at[0, "QPS"] = float(line.split(":")[1])
        elif "number of distances computed" in line:
            df.at[0, "CompsPerQuery"] = float(line.split(":")[2])
    
    # assert if any of the value is empty
    if(df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_nhqkg(df, lines):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if('Time cost:' in line):
            info = line.split(':')
            cons_time = float(info[1])
            df.at[0, "ConstructionTime"] = cons_time
        elif('Search Time' in line):
            info = line.split(' ')
            recall = float(info[5])
            cmps = float(info[9])
            qps = float(info[11])
            df.at[0, "Recall"] = recall
            df.at[0, "CompsPerQuery"] = cmps
            df.at[0, "QPS"] = qps

        # elif('recall@' in line):
        #     info = line.split(':')
        #     recall = float(info[1])
        #     df.at[0, "Recall"] = recall
        # elif "qps" in line:
        #     df.at[0, "QPS"] = float(line.split(":")[1])
        # elif "number of distances computed" in line:
        #     df.at[0, "CompsPerQuery"] = float(line.split(":")[2])
    
    # assert if any of the value is empty
    if(df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1):
        print("error: some value is empty")
        sys.exit(-1)
    return df
    
# def a function that stat total diretion size(MB)
def get_dir_size(dir):
    total = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += get_size(fp)
    return total 

def get_size(file):
    # if is file get size
    if os.path.isfile(file):
        return os.path.getsize(file) / 1024 / 1024
    # if is directory get total size
    elif os.path.isdir(file):
        return get_dir_size(file)
    else:
        return 0

# main 
if __name__ == "__main__":
    res_filename = sys.argv[1]
    dataset = sys.argv[2]
    label_method = sys.argv[3]
    query_method = sys.argv[4]
    distribution = sys.argv[5]
    label_range = int(sys.argv[6])
    query_label_cnt = int(sys.argv[7])
    ef_construction = int(sys.argv[8])
    M = int(sys.argv[9])
    serf_M = int(sys.argv[10])
    K = int(sys.argv[11])
    nprobe = int(sys.argv[12])
    ef_search = int(sys.argv[13])
    threads = int(sys.argv[14])
    gamma = float(sys.argv[15])
    M_beta = int(sys.argv[16])
    alpha = float(sys.argv[17])
    L = int(sys.argv[18])
    partition_size_M = int(sys.argv[19])
    beamsize = int(sys.argv[20])
    split_factor = int(sys.argv[21])
    shift_factor = float(sys.argv[22])
    final_beam_multiply = int(sys.argv[23])
    kgraph_L = int(sys.argv[24])
    iter = int(sys.argv[25])
    S = int(sys.argv[26])
    R = int(sys.argv[27])
    B = float(sys.argv[28])
    kgraph_M = int(sys.argv[29])
    weight_search = float(sys.argv[30])
    dim = int(sys.argv[31])
    N = int(sys.argv[32])
    query_size = int(sys.argv[33])
    acorn_index_file = sys.argv[34]
    diskann_index_label_root = sys.argv[35]
    rfann_index_prefix = sys.argv[36]
    irange_result_file = sys.argv[37]
    rii_index_file = sys.argv[38]
    serf_index_file = sys.argv[39]
    nhq_index_model_file = sys.argv[40]
    nhq_index_attr_file = sys.argv[41]
    ivfpq_index_file = sys.argv[42]
    hnsw_index_file = sys.argv[43]
    vtree_index_prefix = sys.argv[44]
    nhqkg_index_model_file = sys.argv[45]
    nhqkg_index_attr_file = sys.argv[46]
    algo = sys.argv[47]
    output_csv = sys.argv[48]

    df = pd.DataFrame(columns=title_list)
    # create a empty row

    df.at[0, "Paper"] = algo
    df.at[0, "Dataset"] = dataset
    df.at[0, "dim"] = dim
    df.at[0, "N"] = N
    df.at[0, "query_size"] = query_size
    df.at[0, "label_method"] = label_method
    df.at[0, "query_method"] = query_method
    df.at[0, "distribution"] = distribution
    df.at[0, "label_range"] = label_range
    df.at[0, "query_label_cnt"] = query_label_cnt
    df.at[0, "K"] = K
    df.at[0, "Threads"] = threads
    df.at[0, "File"] = res_filename

    if(algo == 'ACORN'):
        df.at[0, "M"] = M
        df.at[0, "ef_construction"] = ef_construction
        df.at[0, "ef_search"] = ef_search
        df.at[0, "gamma"] = gamma
        df.at[0, "M_beta"] = M_beta
        df.at[0, "IndexSize"] = get_size(acorn_index_file)
    elif(algo == 'HNSW'):
        df.at[0, "M"] = M
        df.at[0, "ef_construction"] = ef_construction
        df.at[0, "ef_search"] = ef_search
        df.at[0, "IndexSize"] = get_size(hnsw_index_file)
    elif(algo == 'DiskANN'):
        df.at[0, "M"] = M
        df.at[0, "alpha"] = alpha
        df.at[0, "L"] = L
        df.at[0, "IndexSize"] = get_size(diskann_index_label_root)
    elif(algo == 'SeRF'):
        df.at[0, "serf_M"] = serf_M
        df.at[0, "ef_construction"] = ef_construction
        df.at[0, "ef_search"] = ef_search
        df.at[0, "IndexSize"] = get_size(serf_index_file)
    elif(algo == 'iRangeGraph'):
        df.at[0, "ef_construction"] = ef_construction
        df.at[0, "ef_search"] = ef_search
        df.at[0, "IndexSize"] = get_size(irange_result_file)
    elif(algo == 'WST-opt'):
        df.at[0, "beamsize"] = beamsize
        df.at[0, "split_factor"] = split_factor
        df.at[0, "shift_factor"] = shift_factor
        df.at[0, "final_beam_multiply"] = final_beam_multiply
        df.at[0, "IndexSize"] = get_size(rfann_index_prefix)
    elif(algo == 'Milvus'):
        df.at[0, "nprobe"] = nprobe
    elif(algo == 'RII'):
        df.at[0, "partition_size_M"] = partition_size_M
        df.at[0, "IndexSize"] = get_size(rii_index_file)
    elif(algo == 'IVFPQ'):
        df.at[0, "partition_size_M"] = partition_size_M
        df.ap[0, "nprobe"] = nprobe
        df.at[0, "IndexSize"] = get_size(ivfpq_index_file)
    elif(algo =='NHQ_nsw'):
        df.at[0, "M"] = M
        df.at[0, "ef_construction"] = ef_construction
        df.at[0, "ef_search"] = ef_search
        idx_size1 = get_size(nhq_index_model_file)
        idx_size2 = get_size(nhq_index_attr_file)
        df.at[0, "IndexSize"] = idx_size1 + idx_size2
    elif(algo =='NHQ_kgraph'):
        # kgraph_L iter S R B kgraph_M RANGE(M) PL(ef_cons) weight_search L_search(ef_search)
        df.at[0, "kgraph_L"] = kgraph_L
        df.at[0, "iter"] = iter
        df.at[0, "S"] = S
        df.at[0, "R"] = R
        df.at[0, "B"] = B
        df.at[0, "kgraph_M"] = kgraph_M
        df.at[0, "M"] = M
        df.at[0, "ef_construction"] = ef_construction
        df.at[0, "weight_search"] = weight_search
        df.at[0, "ef_search"] = ef_search
        idx_size1 = get_size(nhqkg_index_model_file)
        idx_size2 = get_size(nhqkg_index_attr_file)
        df.at[0, "IndexSize"] = idx_size1 + idx_size2
    elif(algo == 'Vamana_tree'):
        df.at[0, "beamsize"] = beamsize
        df.at[0, "split_factor"] = split_factor
        df.at[0, "final_beam_multiply"] = final_beam_multiply
        df.at[0, "IndexSize"] = get_size(vtree_index_prefix)

    else:
        print("not support")
        exit()


    df.at[0, "CompsPerQuery"] = -1
    df.at[0, "ConstructionTime"] = -1
    df.at[0, "Recall"] = -1
    df.at[0, "QPS"] = -1
    df.at[0, "CompsPerQuery"] = -1


    # iterate directiary
    for subdir, dirs, files in os.walk(res_filename):
        for file in files:
            # open file as text
            with open(os.path.join(subdir, file), 'r') as f:
                print("file:", os.path.join(subdir, file))
                # find the last substring "Start time"
                lines = f.readlines()
                start_list = []
                for idx, line in enumerate(lines):
                    if "Start time" in line:
                        start_list.append(idx)
                if(algo == "WST-opt"):
                    df = get_info_from_wst(df, lines[start_list[-1]:])
                elif(algo == "IVFPQ"):
                    df = get_info_from_ivfpq(df, lines[start_list[-1]:])
                elif(algo == "HNSW"):
                    df = get_info_from_hnsw(df, lines[start_list[-1]:])
                elif(algo == "Vamana_tree"):
                    df = get_info_from_vtree(df, lines[start_list[-1]:])
                elif(algo == "NHQ_kgraph"):
                    df = get_info_from_nhqkg(df, lines[start_list[-1]:])
                else:
                    print("error: algorithm not supported for dir:", res_filename)
                    sys.exit(-1)
    #print df['Recall']
    print("recall:", df.at[0, 'Recall'])
    # add to csv
    if not os.path.exists(output_csv):
        df.to_csv(output_csv, index=False)
    else:
        # append to csv
        df.to_csv(output_csv, mode='a', header=False, index=False)

                
    
