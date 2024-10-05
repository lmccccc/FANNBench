import numpy as np
import random
import os
from defination import *
import sys
import json



# input: database vector size, query size, attribution range(like [0,1)), query count(like {[0,0.2), [0.3, 0.5))})
def genearte_qrange(method, query_size, attr_range, query_restriction_size, dist, centroid_file=None, query_file=None):
    #generate attribution and query range
    if(method == "float_range"):# float range
        #attr format: random integer from 0 to 300

        #query range format: random range from 0 to 300
        queryattr = np.random.uniform(low=0, high=attr_range, size=(query_size, 2 * query_restriction_size))
        for i in range(query_size):
            rand_size = random.randint(1, query_restriction_size)
            queryattr[i][0 : rand_size * 2] = np.sort(queryattr[i][0 : rand_size * 2])
            queryattr[i][rand_size*2 :] = -1


    elif(method == "keyword"):# keyword
        #key query format: random key from 0 to 30, no more than 5 keys
        if dist == "random":
            queryattr = np.random.randint(0, attr_range, (query_size, 2 * query_restriction_size), dtype='int32')
            for i in range(query_size):
                rand_size = random.randint(1, query_restriction_size)
                queryattr[i][rand_size*2 :] = -1
                # set odd queryattr to even
                queryattr[i][1::2] = queryattr[i][::2]
        elif dist == "in_dist":
            check_file(centroid_file)
            centroids = np.loadtxt(centroid_file)
            query = fvecs_read(query_file)
            qattr = np.zeros((query.shape[0]), dtype='int32')
            for idx, _data in enumerate(query):
                distance = np.linalg.norm(_data - centroids, axis=1)
                min_index = np.where(distance == np.min(distance))[0]
                qattr[idx] = min_index[0]
            queryattr = np.zeros((query_size, 2 * query_restriction_size), dtype='int32')
            for i in range(query_size):
                queryattr[i][0 : 2] = qattr[i]
                queryattr[i][2:] = qattr[i]
        elif dist == "out_dist":
            check_file(centroid_file)
            centroids = np.loadtxt(centroid_file)
            query = fvecs_read(query_file)
            qattr = np.zeros((query.shape[0]), dtype='int32')
            for idx, _data in enumerate(query):
                distance = np.linalg.norm(_data - centroids, axis=1)
                max_index = np.where(distance == np.max(distance))[0]
                qattr[idx] = max_index[0]
            queryattr = np.zeros((query_size, 2 * query_restriction_size), dtype='int32')
            for i in range(query_size):
                queryattr[i][0 : 2] = qattr[i]
                queryattr[i][2:] = qattr[i]

        else:
            print("err no such dist")
            exit()

    # int range, to match diskann that only one label supported, range is [a, a]
    elif(method == "keyword_range"):
        #key query format: random key from 0 to attr_range, no more than query_size keys
        queryattr = np.random.randint(0, attr_range, (query_size, 2 * query_restriction_size), dtype='int32')
        for i in range(query_size):
            # rand_size = random.randint(1, query_restriction_size)
            # queryattr[i][0 : rand_size * 2] = np.sort(queryattr[i][0 : rand_size * 2])
            # queryattr[i][rand_size*2 :] = -1

            queryattr[i][0 : 2] = np.sort(queryattr[i][0 : 2])
            
            # set odd queryattr to even
            # queryattr[i][1::2] = queryattr[i][::2]
    else:
        print("err no such method")
        exit()

    return queryattr


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
        exit()
    return data.shape


def check_data_size(datasetfile, queryfile, attr_size, query_size):
    dataset_n, dataset_d = get_data_size(datasetfile)
    query_n, query_d = get_data_size(queryfile)

    if not query_d == dataset_d:
        print("error query and dataset dim not match")
        exit()
    if not dataset_n == attr_size:
        print("ori db size:", attr_size, " new N: ", dataset_n)
    if not query_n == query_size:
        print("ori query size: ", query_size, " new query size: ", query_n)
    
    return attr_size, query_size

if __name__ == "__main__":
    # ${dataset} 
    # ${N} 
    # ${query_size} 
    # ${dataset_file} 
    # ${query_file} 
    # ${output_dataset_attr_file} 
    # ${output_query_range_file} 
    # ${method}
    if len(sys.argv) < 7 :
        print("error wrong argument")
        exit()
    else:

        query_size = int(sys.argv[1])
        print("query size: ", query_size)
        if not (isinstance(query_size, int) and query_size > 0):
            print("error invalid query size, which should be positive integer")
            exit()
        

        query_file = sys.argv[2]
        print("query file:", query_file)
        check_file(query_file)

        output_query_range_file = sys.argv[3]
        print("query range file:", output_query_range_file)
        check_dir(output_query_range_file)

        method_str = sys.argv[4]
        if(method_str.lower() == "range" or method_str.lower() == "keyword"):
            print("query range generate method: ", method_str)
        else:
            print("can't support such method: ", method_str)

        label_cnt = int(sys.argv[5])
        print("label count: ", label_cnt)

        qlabel_per_query = int(sys.argv[6])
        print("label per query: ", qlabel_per_query)

        dist = sys.argv[7]
        print("distribution: ", dist)

        if dist == "in_dist" or dist == "out_dist":
            centroid_file = sys.argv[8]
            print("centroid file:", centroid_file)
            # directionary check
            # attr_size, query_size = check_data_size(dataset_file, query_file, attr_size, query_size)

            #generate attributions
            queryattr = genearte_qrange(method_str, query_size, label_cnt, qlabel_per_query, dist, centroid_file, query_file)

        else:
            #generate attributions
            queryattr = genearte_qrange(method_str, query_size, label_cnt, qlabel_per_query, dist)
    #write attribution and query range to file
    # write_attr(output_dataset_attr_file, randattr)
    # write_query_range(output_query_range_file, queryattr)

    write_query_range_json(output_query_range_file, queryattr)
    #debug
    
    for i in range(5):
        print("query attr", i, " range=", queryattr[i])