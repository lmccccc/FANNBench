import numpy as np
import sys
import os
import pandas as pd
import re
sys.path.append("..")
from var import Vars
# title = "Paper	Dataset	label method	Threads	dis method	Data size/Query size/dim	K	M(hnsw) R(diskann)		efc(hnsw) beamsize(rfann)	efs  L(diskann)	gamma(acorn) multiply(rfann)	Mbeta	alpha	nprobe	beamSize	final beam multiply	split factor	shift factor	final multiplies	partition size M	recall	QPS	selectivity	ConstructionTime	IndexSize	CompsPerQuery	File"

#algo_list = ["RangeFilteredANN"]

def get_info_from_acorn(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        start_cons = 0
        if(mode == 'construction' and 'Vectors added to hybrid' in line):
            # ('super-postfiltering_2_0.5_1_10_16', 0.6053700000000002, 0.11043572425842285, 380.4889991283417, 2, 7.257514953613281)
            # remove (), split by ','
            cons_time = line.split(' ')[1].rstrip('s')
            df.at[0, "ConstructionTime"] = float(cons_time) - start_cons
        elif('Adding the vectors to the index' in line):
            start_cons = float(line.split(' ')[1].rstrip('s'))
        elif('recall@' in line):
            info = line.split(':')
            recall = float(info[1])
            df.at[0, "Recall"] = recall
        elif "qps" in line:
            df.at[0, "QPS"] = float(line.split(":")[1])
        elif "average distance computations per query" in line:
            df.at[0, "CompsPerQuery"] = float(line.split(":")[1])
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1 or df.at[0, "CompsPerQuery"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_diskann(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for idx, line in enumerate(lines):
        if(mode == 'construction' and 'Indexing time' in line):
            # ('super-postfiltering_2_0.5_1_10_16', 0.6053700000000002, 0.11043572425842285, 380.4889991283417, 2, 7.257514953613281)
            # remove (), split by ','
            cons_time = line.split(':')[1]
            df.at[0, "ConstructionTime"] = float(cons_time)
        elif('Recall@' in line):
            # get next line
            nn_line = lines[idx+2]
            info = nn_line.split()
            print("info:", info)
            recall = float(info[5])
            df.at[0, "Recall"] = recall
            df.at[0, "QPS"] = float(info[1])
            df.at[0, "CompsPerQuery"] = float(info[2])
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1 or df.at[0, "CompsPerQuery"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df



def get_info_from_wst(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if(mode == 'construction' and 'Super optimized postfilter tree build time:' in line):
            # ('super-postfiltering_2_0.5_1_10_16', 0.6053700000000002, 0.11043572425842285, 380.4889991283417, 2, 7.257514953613281)
            # remove (), split by ','
            cons_time = float(line.split(':')[1].split('s')[0])
            df.at[0, "ConstructionTime"] = cons_time
        elif('super-postfiltering' in line):
            # ('super-postfiltering_2_0.5_1_10_16', 0.6053700000000002, 0.11043572425842285, 380.4889991283417, 2, 7.257514953613281)
            # remove (), split by ','
            info = line.split('(')[1].split(')')[0].split(',')
            recall = float(info[1])
            df.at[0, "Recall"] = recall

        elif "qps" in line:
            df.at[0, "QPS"] = float(line.split(":")[1])
        elif "avg comp" in line:
            df.at[0, "CompsPerQuery"] = float(line.split(":")[1])
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1 or df.at[0, "CompsPerQuery"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_vtree(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    # print("lines:", lines)
    for line in lines:
        if(mode == 'construction' and 'Vamana tree build time:' in line):
            print("got line")
            # ('super-postfiltering_2_0.5_1_10_16', 0.6053700000000002, 0.11043572425842285, 380.4889991283417, 2, 7.257514953613281)
            # remove (), split by ','
            cons_time = float(line.split(':')[1].split('s')[0])
            print("line:", line)
            print("cons_time:", cons_time)
            df.at[0, "ConstructionTime"] = cons_time
        elif('vamana-tree' in line):
            # ('super-postfiltering_2_0.5_1_10_16', 0.6053700000000002, 0.11043572425842285, 380.4889991283417, 2, 7.257514953613281)
            # remove (), split by ','
            info = line.split('(')[1].split(')')[0].split(',')
            recall = float(info[2])
            df.at[0, "Recall"] = recall

        elif "qps" in line:
            df.at[0, "QPS"] = float(line.split(":")[1])
        elif "avg comp" in line:
            df.at[0, "CompsPerQuery"] = float(line.split(":")[1])
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1 or df.at[0, "CompsPerQuery"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_ivfpq(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if(mode == 'construction' and 'index construction total cost' in line):
            info = line.split(':')
            cons_time = float(info[1])
            df.at[0, "ConstructionTime"] = cons_time
        elif('recall@' in line):
            info = line.split(':')
            recall = float(info[1])
            df.at[0, "Recall"] = recall
        elif "qps" in line:
            df.at[0, "QPS"] = float(line.split(":")[1])
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_hnsw(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if(mode == 'construction' and 'index construction cost' in line):
            info = line.split(':')
            df.at[0, "ConstructionTime"] = float(info[1])
        elif('recall@' in line):
            info = line.split(':')
            recall = float(info[1])
            df.at[0, "Recall"] = recall
        elif "qps" in line:
            df.at[0, "QPS"] = float(line.split(":")[1])
        elif "number of distances computed" in line:
            df.at[0, "CompsPerQuery"] = float(line.split(":")[2])
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1 or df.at[0, "CompsPerQuery"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_nhqkg(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if(mode == 'construction' and 'Time cost:' in line):
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
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_nhqnsw(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if(mode == 'construction' and 'Build time:' in line):
            info = line.split(':')
            cons_time = float(info[1])
            df.at[0, "ConstructionTime"] = cons_time
        elif('Search Time' in line):
            info = line.split(' ')
            recall = float(info[5])
            cmps = float(info[12])
            qps = float(info[14])
            df.at[0, "Recall"] = recall
            df.at[0, "CompsPerQuery"] = cmps
            df.at[0, "QPS"] = qps
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_rii(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if(mode == 'construction' and 'train time:' in line):
            info = line.split(':')
            cons_time = float(info[-1])
            df.at[0, "ConstructionTime"] = cons_time
        elif('recall@' in line):
            info = line.split(':')
            recall = float(info[1])
            df.at[0, "Recall"] = recall
        elif "qps" in line:
            info = line.split(':')
            df.at[0, "QPS"] = float(info[1])
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_serf(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if(mode == 'construction' and '# Build Index Time:' in line):
            cons_time = float(line.split(':')[1].split('s')[0])
            df.at[0, "ConstructionTime"] = cons_time
        elif('range:' in line):
            # range: 0	 recall: 0.7427	 QPS: 4271	 Comps: 358
            # split by :, \t and space
            
            info = line.split()
            recall = float(info[3])
            qps = float(info[5])
            comps = float(info[7])
            df.at[0, "Recall"] = recall
            df.at[0, "QPS"] = qps
            df.at[0, "CompsPerQuery"] = comps
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1 or df.at[0, "CompsPerQuery"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_milvus(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if(mode == 'construction' and 'insert suc, time cost:' in line): 
            cons_time = float(line.split(':')[1])
            df.at[0, "ConstructionTime"] = cons_time
        elif ('queries cost' in line):
            info = line.split(' ')
            qps = float(info[-1])
            df.at[0, "QPS"] = qps
        elif('recall@' in line):   
            info = line.split(':')
            recall = float(info[-1])
            df.at[0, "Recall"] = recall
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_irange(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for line in lines:
        if(mode == 'construction' and 'construction time' in line):
            cons_time = float(line.split(':')[1].split('s')[0])
            df.at[0, "ConstructionTime"] = cons_time
        elif ('ef:' in line):
            # ef:100, recall:0.99901, qps:3757.52, dco:1289, hop:102, cmp:1302
            info = re.split(r'[ ,:]+', line)
            # print("info:", info)
            qps = float(info[5])
            recall = float(info[3])
            cmps = float(info[-1])
            df.at[0, "QPS"] = qps
            df.at[0, "Recall"] = recall
            df.at[0, "CompsPerQuery"] = cmps
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1 or df.at[0, "CompsPerQuery"] == -1)):
        print("error: some value is empty")
        sys.exit(-1)
    return df

def get_info_from_unify(df, lines, mode):
    #Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery
    for idx, line in enumerate(lines):
        if(mode == 'construction' and 'Index built' in line):
            cons_time = float(line.split("duration:")[1].split(".")[0])
            df.at[0, "ConstructionTime"] = cons_time
        elif ('latency(ms)' in line):
            line = lines[idx+1]
            # ef:100, recall:0.99901, qps:3757.52, dco:1289, hop:102, cmp:1302
            info = line.split()
            # print("info:", info)
            qps = float(info[6])
            recall = float(info[3])
            cmps = float(info[5])
            df.at[0, "QPS"] = qps
            df.at[0, "Recall"] = recall
            df.at[0, "CompsPerQuery"] = cmps
        elif "Maximum resident set size (kbytes)" in line:
            mem = float(line.split(":")[1]) / 1024
            df.at[0, "Memory"] = mem
    # assert if any of the value is empty
    if(mode != 'construction' and (df.at[0, "Recall"] == -1 or df.at[0, "QPS"] == -1 or df.at[0, "CompsPerQuery"] == -1)):
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

# main d
def extract_results(vars:Vars):
    # print("args:", sys.argv)
    assert(vars.algo is not None)

    res_filename = vars.log_file
    dataset = vars.dataset
    label_cnt = vars.label.label_cnt
    query_label = vars.label.query_label
    distribution = vars.label.distribution
    label_range = vars.label.label_range
    query_label_cnt = vars.label.query_label_cnt
    ef_construction = vars.basevars.ef_construction
    M = vars.basevars.M
    serf_M = vars.basevars.serf_M
    K = vars.basevars.K
    nprobe = vars.basevars.nprobe
    ef_search = vars.basevars.ef_search
    threads = vars.basevars.threads
    gamma = vars.basevars.gamma
    M_beta = vars.basevars.M_beta
    alpha = vars.basevars.alpha
    L = vars.basevars.L
    partition_size_M = vars.basevars.partition_size_M
    beamsize = vars.basevars.beamsize
    split_factor = vars.basevars.split_factor
    shift_factor = vars.basevars.shift_factor
    final_beam_multiply = vars.basevars.final_beam_multiply
    kgraph_L = vars.basevars.kgraph_L
    iter = vars.basevars.iter
    S = vars.basevars.S
    R = vars.basevars.R
    B = vars.basevars.B
    kgraph_M = vars.basevars.kgraph_M
    weight_search = vars.basevars.weight_search
    dim = vars.dataset_path.dim
    N = vars.dataset_path.N
    query_size = vars.dataset_path.query_size
    acorn_index_file = vars.algorithm_file.acorn_index_file
    diskann_index_label_root = vars.algorithm_file.diskann_index_label_root
    rfann_index_prefix = vars.algorithm_file.rfann_index_prefix
    irange_result_file = vars.algorithm_file.irange_result_file
    rii_index_file = vars.algorithm_file.rii_index_file
    serf_index_file = vars.algorithm_file.serf_index_file
    nhq_index_model_file = vars.algorithm_file.nhq_index_model_file
    nhq_index_attr_file = vars.algorithm_file.nhq_index_attr_file
    ivfpq_index_file = vars.algorithm_file.ivfpq_index_file
    hnsw_index_file = vars.algorithm_file.hnsw_index_file
    vtree_index_prefix = vars.algorithm_file.vtree_index_prefix
    nhqkg_index_model_file = vars.algorithm_file.nhqkg_index_model_file
    nhqkg_index_attr_file = nhqkg_index_attr_file
    milvus_coll_path = vars.algorithm_file.milvus_coll_path
    algo = vars.algo
    output_csv = vars.result_file
    mode = vars.mode
    B_unify = vars.basevars.B_unify
    AL = vars.basevars.AL
    unify_index_file = vars.algorithm_file.unify_index_file
    Stitched_R = vars.basevars.Stitched_R

    df = pd.DataFrame(columns=vars.basevars.title_list)
    # create a empty row

    df.at[0, "Paper"] = algo
    df.at[0, "Dataset"] = dataset
    df.at[0, "dim"] = dim
    df.at[0, "N"] = N
    df.at[0, "query_size"] = query_size
    df.at[0, "label_cnt"] = label_cnt
    df.at[0, "distribution"] = distribution
    df.at[0, "label_range"] = label_range
    df.at[0, "query_label_cnt"] = query_label_cnt
    df.at[0, "K"] = K
    df.at[0, "Threads"] = threads
    df.at[0, "File"] = res_filename

    if label_cnt > 1 and query_label_cnt == 1:
        df.at[0, "query_label"] = query_label

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
    elif(algo == 'DiskANN_Stitched'):
        df.at[0, "M"] = M
        df.at[0, "alpha"] = alpha
        df.at[0, "L"] = L
        df.at[0, "IndexSize"] = get_size(diskann_index_label_root)
        df.at[0, "Stitched_R"] = Stitched_R
    elif(algo == 'SeRF'):
        df.at[0, "serf_M"] = serf_M
        df.at[0, "ef_construction"] = ef_construction
        df.at[0, "ef_search"] = ef_search
        df.at[0, "IndexSize"] = get_size(serf_index_file)
    elif(algo == 'iRangeGraph'):
        df.at[0, "ef_construction"] = ef_construction
        df.at[0, "ef_search"] = ef_search
        df.at[0, "IndexSize"] = get_size(irange_result_file)
    elif(algo == 'WST_opt'):
        df.at[0, "beamSize"] = beamsize
        df.at[0, "split_factor"] = split_factor
        df.at[0, "shift_factor"] = shift_factor
        df.at[0, "final_beam_multiply"] = final_beam_multiply
        df.at[0, "IndexSize"] = get_size(rfann_index_prefix)
    elif(algo == 'Milvus_IVFPQ'):
        df.at[0, "nprobe"] = nprobe
        df.at[0, "IndexSize"] = -1
        # df.at[0, "IndexSize"] = get_size(milvus_coll_path)
    elif(algo == 'Milvus_HNSW'):
        df.at[0, "M"] = M
        df.at[0, "ef_construction"] = ef_construction
        df.at[0, "ef_search"] = ef_search
        df.at[0, "IndexSize"] = -1
        # df.at[0, "IndexSize"] = get_size(milvus_coll_path)
    elif(algo == 'RII'):
        df.at[0, "partition_size_M"] = partition_size_M
        df.at[0, "IndexSize"] = get_size(rii_index_file)
    elif(algo == 'IVFPQ'):
        df.at[0, "partition_size_M"] = partition_size_M
        df.at[0, "nprobe"] = nprobe
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
        df.at[0, "beamSize"] = beamsize
        df.at[0, "split_factor"] = split_factor
        df.at[0, "final_beam_multiply"] = final_beam_multiply
        df.at[0, "IndexSize"] = get_size(vtree_index_prefix)
    elif(algo == 'UNIFY'):
        df.at[0, "ef_construction"] = ef_construction
        df.at[0, "ef_search"] = ef_search
        df.at[0, "AL"] = AL
        df.at[0, "M"] = M
        df.at[0, "B_unify"] = B_unify
        df.at[0, "IndexSize"] = get_size(unify_index_file)

        
    else:
        print("not support")
        exit()


    df.at[0, "CompsPerQuery"] = -1
    df.at[0, "ConstructionTime"] = -1
    df.at[0, "Recall"] = -1
    df.at[0, "QPS"] = -1
    df.at[0, "CompsPerQuery"] = -1


    # iterate directiary
    # for subdir, dirs, files in os.walk(res_filename):
    #     for file in files:
            # open file as text
    with open(res_filename, 'r') as f:
        print("file:", res_filename)
        # find the last substring "Start time"
        lines = f.readlines()
        start_list = []
        for idx, line in enumerate(lines):
            if "Start time" in line:
                start_list.append(idx)
        if(algo == "WST_opt"):
            df = get_info_from_wst(df, lines[start_list[-1]:], mode)
        elif(algo == "ACORN"):
            df = get_info_from_acorn(df, lines[start_list[-1]:], mode)
        elif(algo == 'DiskANN'):
            df = get_info_from_diskann(df, lines[start_list[-1]:], mode)
        elif(algo == "IVFPQ"):
            df = get_info_from_ivfpq(df, lines[start_list[-1]:], mode)
        elif(algo == "HNSW"):
            df = get_info_from_hnsw(df, lines[start_list[-1]:], mode)
        elif(algo == "Vamana_tree"):
            df = get_info_from_vtree(df, lines[start_list[-1]:], mode)
        elif(algo == "NHQ_kgraph"):
            df = get_info_from_nhqkg(df, lines[start_list[-1]:], mode)
        elif(algo == "NHQ_nsw"):
            df = get_info_from_nhqnsw(df, lines[start_list[-1]:], mode)
        elif(algo == "RII"):
            df = get_info_from_rii(df, lines[start_list[-1]:], mode)
        elif(algo == "SeRF"):
            df = get_info_from_serf(df, lines[start_list[-1]:], mode)
        elif(algo == 'Milvus_IVFPQ'):
            df = get_info_from_milvus(df, lines[start_list[-1]:], mode)
        elif(algo == 'Milvus_HNSW'):
            df = get_info_from_milvus(df, lines[start_list[-1]:], mode)
        elif(algo == 'iRangeGraph'):
            df = get_info_from_irange(df, lines[start_list[-1]:], mode)
        elif(algo == 'UNIFY'):
            df = get_info_from_unify(df, lines[start_list[-1]:], mode)
        else:
            print("error: algorithm not supported for dir:", res_filename)
            sys.exit(-1)
    #print df['Recall']

    if mode != 'construction':
        print("recall:", df.at[0, 'Recall'], ", qps:", df.at[0, 'QPS'], ", cmps:", df.at[0, 'CompsPerQuery'])
    else:
        print("construction time:", df.at[0, 'ConstructionTime'], ", index size:", df.at[0, 'IndexSize'])

    # add to csv
    import fcntl
    with open(output_csv, 'a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            if not os.path.exists(output_csv):
                df.to_csv(output_csv, index=False)
            else:
                # append to csv
                df.to_csv(output_csv, mode='a', header=False, index=False)
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)

                
    
