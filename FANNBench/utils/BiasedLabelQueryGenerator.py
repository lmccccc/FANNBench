import numpy as np
import random
import os
from defination import *
import sys
import json
import math


def genearte_query(attr_cnt, query_size, attr_range, query_attr_size, distribution, query_attr, N, attr_file, centroid_file, query_file, ori_query_file, ori_query_size):
    #generate attribution and query range
    
    
    print("select biased vector for biased label query to fit in-/out- distribution")
    if not attr_cnt == 1:
        print("error, only support one attr")
        sys.exit(-1)
    if not (distribution == "in_dist" or distribution == "out_dist"):
        print("error, only support in_dist and out_dist")
        sys.exit(-1)
    if not query_attr_size == 1:
        print("error, only support query_attr_size=1")
        sys.exit(-1)
    if not query_attr > 0:
        print("error, query_attr should be 1~19")
        sys.exit(-1)

    check_file(centroid_file)
    centroids = np.loadtxt(centroid_file)
    query = fvecs_read(ori_query_file)
    
    target_centroid = centroids[query_attr]
    distance = np.linalg.norm(query - target_centroid, axis=1)
    sorted_index = np.argsort(distance)
    target_query = query[sorted_index[:query_size]]
    return target_query


def write_attr_json(filepath, attr):
    if not ".json" in filepath:
        print("error, json should be stored in .json file, not ", filepath)
    with open(filepath, 'w') as file:
        json.dump(attr.reshape(-1).tolist(), file, indent=4)
    

def write_query_range_json(filepath, attr):
    if not ".json" in filepath:
        print("error, json should be stored in .json file, not ", filepath)
    with open(filepath, 'w') as file:
        json.dump(attr.reshape(-1).tolist(), file, indent=4)

def get_data_size(file):
    if("ivecs" in file):
        data = ivecs_read(file)
    elif("fvecs" in file):
        data = fvecs_read(file)
    elif("bvecs" in file):
        data = bvecs_read(file)
    else:
        print ("cannot support such file type:", file)
        sys.exit()
    return data.shape


def check_data_size(datasetfile, queryfile, attr_size, query_size):
    dataset_n, dataset_d = get_data_size(datasetfile)
    query_n, query_d = get_data_size(queryfile)

    if not query_d == dataset_d:
        print("error query and dataset dim not match")
        sys.exit()
    if not dataset_n == attr_size:
        print("ori db size:", attr_size, " new N: ", dataset_n)
    if not query_n == query_size:
        print("ori query size: ", query_size, " new query size: ", query_n)
    
    return attr_size, query_size

if __name__ == "__main__":
    if len(sys.argv) < 8 :
        print("error wrong argument")
        sys.exit()
    else:

        query_size = int(sys.argv[1])
        print("query size: ", query_size)
        if not (isinstance(query_size, int) and query_size > 0):
            print("error invalid query size, which should be positive integer")
            sys.exit()
        

        query_file = sys.argv[2]
        print("query file:", query_file)

        output_query_range_file = sys.argv[3]
        print("query range file:", output_query_range_file)
        check_dir(output_query_range_file)

        attr_cnt = int(sys.argv[4])
        print("attr count: ", attr_cnt)

        attr_range = int(sys.argv[5])
        print("attr range count: ", attr_range)

        query_attr_size = int(sys.argv[6])
        print("query attr count: ", query_attr_size)

        distribution = sys.argv[7]
        print("distributionribution: ", distribution)

        query_attr = int(sys.argv[8])
        print("query attr: ", query_attr)

        centroid_file = sys.argv[9]

        N = int(sys.argv[10])
        print("N: ", N)

        attr_file = sys.argv[11]
        print("attr file:", attr_file)

        
        ori_query_file = sys.argv[13]
        print("ori query file:", ori_query_file)
        check_file(ori_query_file)

        ori_query_size = int(sys.argv[14])
        print("ori query size: ", ori_query_size)
            

        if distribution == "in_dist" or distribution == "out_dist":
            print("centroid file:", centroid_file)
            # directionary check
            # attr_size, query_size = check_data_size(dataset_file, query_file, attr_size, query_size)

            #generate attributions
            query = genearte_query(attr_cnt, query_size, attr_range, query_attr_size, distribution, query_attr, N, attr_file, centroid_file, query_file, ori_query_file, ori_query_size)

        else:
            print("error, only support categorical biased query")
            sys.exit(-1)
        

    fvecs_write(query_file, query)
    print("query size: ", query.shape)
    print("write biased query to ", query_file)

    