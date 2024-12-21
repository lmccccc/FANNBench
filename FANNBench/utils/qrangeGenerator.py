import numpy as np
import random
import os
from defination import *
import sys
import json
import math


'''
attr_cnt > 1: keyword query(more than one attr for each vector)
    random: unit distributionribution
        query_attr_size = 1: generate one query attr defined by "query_attr", used for certain key query. suitable for diskann, etc.
        query_attr_size = attr_cnt: generate multi attr for one query, size="query_attr_size", used for nhq.
    in_dist/out_dist:
        query_attr_size = 1: generate one query attr, based on the distance between query and centroid. used for diskann, etc.
        query_attr_size = attr_cnt: generate multi attr for one query, based on the distance between query and centroid, size="query_attr_size", used for nhq.

attr_cnt = 1: range query
    random: unit distributionribution, range=[random, random+query_attr_size-1].
    in_dist/out_dist: generate one range query, range=[random-query_attr_size/2, random+query_attr_size/2], based on the distance between query and centroid.
'''
def genearte_qrange(attr_cnt, query_size, attr_range, query_attr_size, distribution, query_attr, centroid_file=None, query_file=None):
    #generate attribution and query range
    if(attr_cnt > 1):# keyword query. for each vector may have more than one attr. So query should be only one fixed attr
        #attr format: random integer from 0 to 300
        if distribution == "random":
            if query_attr_size == 1:
                #query range format: random range from 0 to 300
                queryattr = np.full((query_size, 2), query_attr)
            elif query_attr_size == attr_cnt: # nhq
                queryattr = np.random.randint(0, attr_range, (query_size, query_attr_size), dtype='int32')
            else:
                print("error, query_attr_size should be 1 or attr_cnt")
        else:
            check_file(centroid_file)
            centroids = np.loadtxt(centroid_file)
            query = fvecs_read(query_file)
            qattr = np.zeros((query.shape[0]), dtype='int32')
            elements = [i for i in range(attr_range)]

            if query_attr_size == 1:


                if distribution == "in_dist":
                    for idx, _data in enumerate(query):
                        distance = np.linalg.norm(_data - centroids, axis=1)
                        dis_sum = np.sum(distance)
                        probabilities = 1 - distance/dis_sum
                        probabilities = probabilities
                        selected_element = random.choices(elements, weights=probabilities, k=1)
                        qattr[idx] = elements.index(selected_element[0])

                elif distribution == "out_dist":
                    for idx, _data in enumerate(query):
                        distance = np.linalg.norm(_data - centroids, axis=1)
                        dis_sum = np.sum(distance)
                        probabilities = distance/dis_sum
                        probabilities = probabilities
                        selected_element = random.choices(elements, weights=probabilities, k=1)
                        qattr[idx] = elements.index(selected_element[0])
                
                queryattr = np.zeros((query_size, 2), dtype='int32')
                for i in range(query_size):
                    queryattr[i][0 : 2] = qattr[i]
                    queryattr[i][2:] = qattr[i]

            elif query_attr_size == attr_cnt: # nhq
                if distribution == "in_dist":
                    for idx, _data in enumerate(query):
                        distance = np.linalg.norm(_data - centroids, axis=1)
                        dis_sum = np.sum(distance)
                        probabilities = 1 - distance/dis_sum
                        probabilities = probabilities
                        selected_element = random.choices(elements, weights=probabilities, k=query_attr_size)
                        qattr[idx] = elements.index(selected_element)

                elif distribution == "out_dist":
                    for idx, _data in enumerate(query):
                        distance = np.linalg.norm(_data - centroids, axis=1)
                        dis_sum = np.sum(distance)
                        probabilities = distance/dis_sum
                        probabilities = probabilities
                        selected_element = random.choices(elements, weights=probabilities, k=query_attr_size)
                        qattr[idx] = elements.index(selected_element)
                
                queryattr = qattr
            else:
                print("error, query_attr_size should be 1 or attr_cnt")



    elif(attr_cnt == 1):# range query. For each vector with only one attr.
        #key query format: random key from 0 to 30, no more than 5 keys
        if distribution == "random":
            queryattr = np.random.randint(0, attr_range-query_attr_size+1, (query_size, 2), dtype='int32') # [0, attr_range-query_attr_size+1)
            queryattr[:][1::2] = queryattr[:][::2]+query_attr_size-1 # [x, x+query_attr_size]
        else:
            check_file(centroid_file)
            centroids = np.loadtxt(centroid_file)
            query = fvecs_read(query_file)
            qattr = np.zeros((query.shape[0]), dtype='int32')
            

            centroid_size = min(128, attr_range)
            segment_size = math.ceil(attr_range / centroid_size)
            elements = [i for i in range(centroid_size)]
            element_vals = [elements[i]*segment_size for i in range(centroid_size)]
            if distribution == "in_dist":
                for idx, _data in enumerate(query):
                    distance = np.linalg.norm(_data - centroids, axis=1)
                    dis_sum = np.sum(distance)
                    probabilities = 1 - distance/dis_sum
                    selected_element = random.choices(elements, weights=probabilities, k=1)[0]


                    if attr_range <=128:
                        qattr[idx] = selected_element
                    else:
                        # generate random int by normal distribution (selected_element[0], attr_range/centroid_size)
                        mean = element_vals[selected_element] + segment_size // 2
                        var = attr_range/centroid_size
                        while True:
                            qattr[idx] = int(np.random.normal(mean, var))
                            if qattr[idx] >= 0 and qattr[idx] < attr_range:
                                break

                queryattr = np.zeros((query_size, 2), dtype='int32')
                if(query_attr_size == 1):
                    for i in range(query_size):
                        queryattr[i][0 : 2] = qattr[i]
                        queryattr[i][2:] = qattr[i]
                else:
                    half_range = query_attr_size // 2
                    for i in range(query_size):
                        queryattr[i][0] = qattr[i] - half_range
                        if queryattr[i][0] < 0:
                            queryattr[i][0] = 0
                        queryattr[i][1] = qattr[i] + half_range
                        if queryattr[i][1] >= attr_range:
                            queryattr[i][1] = attr_range - 1

            elif distribution == "out_dist":
                for idx, _data in enumerate(query):
                    distance = np.linalg.norm(_data - centroids, axis=1)

                    dis_sum = np.sum(distance)
                    probabilities = distance/dis_sum
                    selected_element = random.choices(elements, weights=probabilities, k=1)
                    qattr[idx] = elements.index(selected_element[0])

                    # max_index = np.where(distance == np.max(distance))[0]
                    # qattr[idx] = max_index[0]
                queryattr = np.zeros((query_size, 2), dtype='int32')
                for i in range(query_size):
                    queryattr[i][0 : 2] = qattr[i]
                    queryattr[i][2:] = qattr[i]

            else:
                print("err no such distribution")
                exit()
        

        



    else:
        print("err no support such attr_cnt:", attr_cnt)
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

        attr_cnt = int(sys.argv[4])
        print("attr count: ", attr_cnt)

        attr_range = int(sys.argv[5])
        print("attr count: ", attr_range)

        query_attr_size = int(sys.argv[6])
        print("query attr count: ", query_attr_size)

        distribution = sys.argv[7]
        print("distributionribution: ", distribution)

        query_attr = int(sys.argv[8])
        print("query attr: ", query_attr)

        if distribution == "in_dist" or distribution == "out_dist":
            centroid_file = sys.argv[9]
            print("centroid file:", centroid_file)
            # directionary check
            # attr_size, query_size = check_data_size(dataset_file, query_file, attr_size, query_size)

            #generate attributions
            queryattr = genearte_qrange(attr_cnt, query_size, attr_range, query_attr_size, distribution, query_attr, centroid_file, query_file)

        else:
            #generate attributions
            queryattr = genearte_qrange(attr_cnt, query_size, attr_range, query_attr_size, distribution, query_attr)
    #write attribution and query range to file
    # write_attr(output_dataset_attr_file, randattr)
    # write_query_range(output_query_range_file, queryattr)

    write_query_range_json(output_query_range_file, queryattr)
    #debug
    
    for i in range(5):
        print("query attr", i, " range=", queryattr[i])
    
    # plot query attr distribution
    import matplotlib.pyplot as plt
    plt.hist(queryattr.reshape(-1), bins=min(128, attr_range))
    plt.show()


    