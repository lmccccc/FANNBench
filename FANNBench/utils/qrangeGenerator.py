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
def genearte_qrange(attr_cnt, query_size, attr_range, query_attr_size, distribution, query_attr, N, attr_file, centroid_file=None, query_file=None):
    #generate attribution and query range
    if(attr_cnt > 1):# keyword query. for each vector may have more than one attr. So query should be only one fixed attr
        pass
        # #attr format: random integer from 0 to attr_range
        # if distribution == "random":
        #     if query_attr_size == 1:
        #         #query range format: random range from 0 to attr_range
        #         queryattr = np.full((query_size, 2), query_attr)
        #     elif query_attr_size == attr_cnt: # nhq
        #         queryattr = np.full((query_size, query_attr_size), query_attr)
        #     else:
        #         print("error, query_attr_size should be 1 or attr_cnt")
        # else:
        #     check_file(centroid_file)
        #     centroids = np.loadtxt(centroid_file)
        #     query = fvecs_read(query_file)
        #     qattr = np.zeros((query.shape[0]), dtype='int32')
        #     elements = [i for i in range(attr_range)]

        #     if query_attr_size == 1:


        #         if distribution == "in_dist":
        #             for idx, _data in enumerate(query):
        #                 distance = np.linalg.norm(_data - centroids, axis=1)
        #                 dis_sum = np.sum(distance)
        #                 probabilities = 1 - distance/dis_sum
        #                 probabilities = probabilities
        #                 selected_element = random.choices(elements, weights=probabilities, k=1)
        #                 qattr[idx] = elements.index(selected_element[0])

        #         elif distribution == "out_dist":
        #             for idx, _data in enumerate(query):
        #                 distance = np.linalg.norm(_data - centroids, axis=1)
        #                 dis_sum = np.sum(distance)
        #                 probabilities = distance/dis_sum
        #                 probabilities = probabilities
        #                 selected_element = random.choices(elements, weights=probabilities, k=1)
        #                 qattr[idx] = elements.index(selected_element[0])
                
        #         queryattr = np.zeros((query_size, 2), dtype='int32')
        #         for i in range(query_size):
        #             queryattr[i][0 : 2] = qattr[i]
        #             queryattr[i][2:] = qattr[i]

        #     elif query_attr_size == attr_cnt: # nhq
        #         print("qrange for nhq")
        #         if distribution == "in_dist":
        #             for idx, _data in enumerate(query):
        #                 distance = np.linalg.norm(_data - centroids, axis=1)
        #                 dis_sum = np.sum(distance)
        #                 probabilities = 1 - distance/dis_sum
        #                 probabilities = probabilities
        #                 selected_element = random.choices(elements, weights=probabilities, k=query_attr_size)
        #                 qattr[idx] = elements.index(selected_element)

        #         elif distribution == "out_dist":
        #             for idx, _data in enumerate(query):
        #                 distance = np.linalg.norm(_data - centroids, axis=1)
        #                 dis_sum = np.sum(distance)
        #                 probabilities = distance/dis_sum
        #                 probabilities = probabilities
        #                 selected_element = random.choices(elements, weights=probabilities, k=query_attr_size)
        #                 qattr[idx] = elements.index(selected_element)
        #         else:
        #             queryattr = np.full((query_size, query_attr_size), query_attr)
                
        #         queryattr = qattr
        #     else:
        #         print("error, query_attr_size should be 1 or attr_cnt")
        #         exit(-1)



    elif(attr_cnt == 1):# For each vector with only one attr.

        range_map = {1:1, 2:0.9, 3:0.8, 4:0.7, 5:0.6, 6:0.5, 7:0.4, 8:0.3, 9:0.2, 10:0.1, \
                        11:0.09, 12:0.08, 13:0.07, 14:0.06, 15:0.05, 16:0.04, 17:0.03, 18:0.02, \
                        19:0.01, 20:0.001}
        if distribution == "random" or distribution == "real":
            if query_attr == 0: # range query
                if not query_attr_size in range_map.keys():
                    print("error, query_attr_size should be in range_map:", range_map.keys())
                    exit(-1)
                selectivity = range_map[query_attr_size]
                print("selectivity:", selectivity)
                query_nbr = int(N * selectivity)
                print("N:", N, "query range cnt:", query_nbr)
                attr = np.array(json.load(open(attr_file)))
                # sort index by attr
                sorted_index = np.argsort(attr)
                # generate random query range
                q_ordered_range = np.random.randint(0, N-query_nbr, (query_size, 2), dtype='int32')
                q_ordered_range[:, 1] = q_ordered_range[:, 0] + query_nbr # [x, x+query_attr_size]
                # convert query index range to attr range
                q_idx = sorted_index[q_ordered_range]
                queryattr = attr[q_idx]

                #check selectivity
                list_qr = queryattr.reshape(-1).tolist()
                for i in range(5):
                    print("attr ", i, ": ", attr[i])
                for i in range(5):
                    left = list_qr[i*2]
                    right = list_qr[i*2+1]
                    valid = 0
                    for j in range(N):
                        if attr[j] >= left and attr[j] <= right:
                            valid += 1
                    print(i, " query range:", left, right, " valid cnt:", valid, " selectivity:", valid/N)



                return queryattr
            else: # label query
                assert(distribution != "real")
                print("generate for categorical query, label:", query_attr)
                queryattr = np.full((query_size, 2), query_attr)
                return queryattr
        else:
            check_file(centroid_file)
            centroids = np.loadtxt(centroid_file)
            query = fvecs_read(query_file)
            qattr = np.zeros((query_size, 2), dtype='int32')
            


            centroid_size = min(128, attr_range)
            segment_size = math.ceil(attr_range / centroid_size)
            elements = [i for i in range(centroid_size)]
            element_vals = [elements[i]*segment_size for i in range(centroid_size)]
            
            if not query_attr_size in range_map.keys():
                print("error, query_attr_size should be in range_map:", range_map.keys())
                exit(-1)

            selectivity = range_map[query_attr_size]

            print("selectivity:", selectivity)
            query_nbr = N * selectivity
            attr = np.array(json.load(open(attr_file)))
            sorted_index = np.argsort(attr)
            sorted_attr = attr[sorted_index]
            element_idx = [np.searchsorted(sorted_attr, element_vals[i]) for i in range(centroid_size)]
            shift_range = query_nbr // 10

            print("generate qrange for biased label")
            # generate q range for real label
            if query_attr == 0: # range query
                if distribution == "in_dist":
                    for idx, _data in enumerate(query):
                        distance = np.linalg.norm(_data - centroids, axis=1)
                        # get idx of max distance
                        selected_element = np.argmax(distance)

                        # if attr_range <=128:
                        #     qattr[idx] = selected_element
                        # else:

                        # generate random int by normal distribution (selected_element[0], attr_range/centroid_size
                        central_idx = element_idx[selected_element]
                        random_shift = int(np.random.normal(0, shift_range))
                        start = max(0, central_idx + random_shift - query_nbr//2)
                        start = min(start, N - 1 - query_nbr)
                        qattr[idx][0] = start
                        qattr[idx][1] = start + query_nbr

                    # if(query_attr_size == 1): # label query
                        # for i in range(query_size):
                        #     queryattr[i][0 : 2] = qattr[i]
                        #     queryattr[i][2:] = qattr[i]
                    # else: # range query
                        # half_range = query_attr_size // 2
                        # for i in range(query_size):
                        #     queryattr[i][0] = qattr[i] - half_range
                        #     if queryattr[i][0] < 0:
                        #         queryattr[i][0] = 0
                        #     queryattr[i][1] = qattr[i] + half_range
                        #     if queryattr[i][1] >= attr_range:
                        #         queryattr[i][1] = attr_range - 1
                    queryattr = attr[sorted_index[qattr]]
                    return queryattr

                elif distribution == "out_dist":
                    for idx, _data in enumerate(query):
                        distance = np.linalg.norm(_data - centroids, axis=1)
                        selected_element = np.argmin(distance)
                        central_idx = element_idx[selected_element]
                        random_shift = int(np.random.normal(0, shift_range))
                        start = max(0, central_idx + random_shift - query_nbr//2)
                        start = min(start, N - 1 - query_nbr)
                        qattr[idx][0] = start
                        qattr[idx][1] = start + query_nbr

                else:
                    print("err no such distribution")
                    exit()
            else: # label query
                assert(distribution != "real")
                print("generate for categorical query, label:", query_attr)
                queryattr = np.full((query_size, 2), query_attr)
                return queryattr
        

        



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
    if len(sys.argv) < 8 :
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

            

        if distribution == "in_dist" or distribution == "out_dist":
            print("centroid file:", centroid_file)
            # directionary check
            # attr_size, query_size = check_data_size(dataset_file, query_file, attr_size, query_size)

            #generate attributions
            queryattr = genearte_qrange(attr_cnt, query_size, attr_range, query_attr_size, distribution, query_attr, N, attr_file, centroid_file, query_file)

        else:
            #generate attributions
            queryattr = genearte_qrange(attr_cnt, query_size, attr_range, query_attr_size, distribution, query_attr, N, attr_file)
        
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


    