import numpy as np
import random
import os
from defination import *
import sys
import json
import time
import math

def generate_zipf_attrs(num_attrs, zipf_exponent=1.2):
    """
    Generate attrs based on a Zipf distribution.

    :param num_attrs: Number of distinct attrs.
    :param num_samples: Number of samples to generate.
    :param zipf_exponent: The exponent parameter for Zipf's distribution. Default is 1.0.
    :return: List of generated attrs.
    """
    # Generate ranks (1, 2, ..., num_attrs)
    # ranks = np.arange(1, num_attrs + 1)

    # Calculate probabilities using Zipf's law: p(k) ∝ 1 / k^s
    # probabilities = 1 / (ranks ** zipf_exponent)
    probabilities = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])

    # Normalize the probabilities to sum to 1
    # probabilities /= probabilities.sum()
    generated_attrs = []
    while(len(generated_attrs) == 0):
        for i in range(num_attrs):
            val = random.random()
            if val <= probabilities[i]:
                generated_attrs.append(i)
    return generated_attrs

# input: database vector size, query size, attribution range(like [0,1)), query count(like {[0,0.2), [0.3, 0.5))})
def genearte_attr(db_size, attr_cnt, attr_range, distribution, query_attr_cnt, query_label, data=None, train=None, centroid_file=None):
    if(attr_range == -1):
        attr_range = 2**31 - 1
    
    if(distribution == "random"):
        print("generating for dist:", distribution)
        if(attr_cnt == 1 and query_label == 0):
            # numberical
            attr = np.random.randint(0, attr_range, db_size, dtype='int32').reshape(db_size).tolist()
        elif (attr_cnt == 1 and query_label > 0):
            print("generate single label for label filtering methods")
            # categorical nhq and filtered diskann
            prob_map = {1:1, 2:0.9, 3:0.8, 4:0.7, 5:0.6, 6:0.5, 7:0.4, 8:0.3, 9:0.2, 10:0.1, 11:0.09, 12:0.08, 13:0.07, 14:0.06, 15:0.05, 16:0.04, 17:0.03, 18:0.02, 19:0.01, 20:0.001}
            prob = prob_map[query_label]
            # print("label:", query_label, " prob:", prob)
            # assign 6, 10, 19, 20
            support_query_label = [6, 10, 19, 20]
            prob_list = np.array([prob_map[val] for val in support_query_label])
            prob_sum = [np.sum(prob_list[:i]) for i in range(1, prob_list.shape[0]+1)]
            print("support query label:", support_query_label)
            print("corresponding selectivity:", prob_list)
            print("prob_sum:", prob_sum)
            attr = np.zeros(db_size, dtype='int32').reshape(db_size)
            if query_label not in support_query_label:
                print("error only support query label:", support_query_label)
                sys.exit(-1)
            if prob_sum[-1] > 1:
                print("error too much labels to assign")
                sys.exit(-1)

            stat = np.zeros(len(support_query_label), dtype='int32')
            for i in range(db_size):
                val = random.random()
                for idx, prob in enumerate(prob_sum):
                    if val < prob_sum[idx]:
                        attr[i] = support_query_label[idx]
                        stat[idx] += 1
                        break
                    attr[i] = np.random.randint(low=0, high=attr_range)
                    while attr[i] in support_query_label:
                        attr[i] = np.random.randint(low=0, high=attr_range)
            print("selectivity:", stat/db_size)
            attr = attr.tolist()
            return attr


        # elif(attr_cnt > 1 and query_attr_cnt == 1):# keywords
        #     attr = []
        #     zipf_exponent = 1.2
        #     print("generating zipf attrs")
        #     for data_ind in range(db_size):
        #         attrs = generate_zipf_attrs(attr_cnt, zipf_exponent)
        #         attr.append(attrs)
        # elif(attr_cnt > 1 and query_attr_cnt > 1):# multi-attr nhq
        #     assert(attr_cnt == query_attr_cnt)
        #     prob_map = {1:1, 2:0.9, 3:0.8, 4:0.7, 5:0.6, 6:0.5, 7:0.4, 8:0.3, 9:0.2, 10:0.1, 11:0.09, 12:0.08, 13:0.07, 14:0.06, 15:0.05, 16:0.04, 17:0.03, 18:0.02, 19:0.01}
        #     prob = prob_map[query_label]
        #     attr = np.zeros((db_size, attr_cnt), dtype='int32')
        #     # print("size:", db_size, attr_cnt)
        #     print(attr.shape)
        #     for i in range(db_size):
        #         val = random.random()
        #         if val <= prob:
        #             attr[i, :] = query_label
        #         else:
        #             attr[i, :] = np.random.randint(low=0, high=attr_range, size=(1, attr_cnt))
        #     attr = attr.tolist()
        #     print("attr shape:", len(attr), len(attr[0]))

            
        # int range, to match diskann that only one attr supported, range is [a, a]
        else:
            print("err no such attr_cnt:", attr_cnt)
            exit()
    elif(distribution == "in_dist" or distribution == "out_dist"):
        from sklearn.cluster import KMeans
        assert train is not None
        assert data is not None
        assert centroid_file is not None

        
        centroid_size = min(128, attr_range)
        if os.path.exists(centroid_file):
            print("load centroids from file:", centroid_file)
            centroids = np.loadtxt(centroid_file)
        else:
            print("generating for dist:", distribution)
            t0 = time.time()
            kmeans = KMeans(n_clusters=centroid_size, random_state=42)
            kmeans.fit(train)
            t1 = time.time()
            print("kmeans time:", t1-t0)
            print("generating centroids")
            centroids = kmeans.cluster_centers_
            np.savetxt(centroid_file, centroids, delimiter=' ', fmt='%f')
            print("save centroids to file:", centroid_file)
        # write centroids into file

        print("data shape:", data.shape)
        print("centroids shape:", centroids.shape)

        labels = np.random.randint(low=0, high=attr_range, size=(db_size), dtype='int32')
        label_ordered_idx = np.argsort(labels)
        labels = labels[label_ordered_idx]
        attr = np.zeros((db_size), dtype='int32')

        if(attr_cnt == 1 and query_label == 0): # range query attrs

            elements = [i for i in range(centroid_size)]
            segment_size = math.ceil(attr_range / centroid_size)
            centroid_vals = [segment_size * i + round(segment_size / 2) for i in range(centroid_size)]
            print("centroid_vals:", centroid_vals)
            print("centroid_size:", len(centroid_vals))
            
            clusters = [[] for i in range(centroid_size)]
            for idx, _data in enumerate(data):
                distance = np.linalg.norm(_data - centroids, axis=1)
                dis_sum = np.sum(distance)
                probabilities = 1 - distance/dis_sum
                selected_element = random.choices(elements, weights=probabilities, k=1)[0]
                clusters[selected_element].append(idx)
                # if attr_range <=128:
                #     attr[idx] = selected_element
                # else:
                #     # generate random int by normal distribution (selected_element[0], attr_range/centroid_size)
                #     mean = centroid_vals[selected_element]
                #     var = attr_range/centroid_size
                #     while True:
                #         attr[idx] = int(np.random.normal(mean, var))
                #         if attr[idx] >= 0 and attr[idx] < attr_range:
                            # break
            # split data
            offset = [0]
            for i in range(centroid_size):
                offset.append(offset[i] + len(clusters[i]))
            for i in range(centroid_size):
                start = offset[i]
                end = offset[i+1]
                attr[np.array(clusters[i])] = labels[start:end]
            attr = attr.tolist()
            return attr

        
        elif (attr_cnt == 1 and query_label > 0):
            print("generate single label for label filtering methods")
            # nhq and filtered diskann
            prob_map = {1:1, 2:0.9, 3:0.8, 4:0.7, 5:0.6, 6:0.5, 7:0.4, 8:0.3, 9:0.2, 10:0.1, 11:0.09, 12:0.08, 13:0.07, 14:0.06, 15:0.05, 16:0.04, 17:0.03, 18:0.02, 19:0.01, 20:0.001}
            prob = prob_map[query_label]
            print("label:", query_label, " prob:", prob)
            if query_label == 1:
                attr = np.full((db_size), query_label, dtype='int32').tolist()
            else:
                attr = np.zeros((db_size), dtype='int32')
                target_centroid = centroids[query_label]
                distance = np.linalg.norm(data - target_centroid, axis=1)
                top_dis_idx = np.argsort(distance)

                attr[top_dis_idx[: int(db_size*prob)]] = query_label
                rest_idx = top_dis_idx[int(db_size*prob) :]
                for idx in rest_idx:
                    while True:
                        attr[idx] = np.random.randint(low=0, high=attr_range)
                        if attr[idx] != query_label:
                            break
            attr = attr.tolist()
            return attr
        

        # elif(attr_cnt > 1 and query_attr_cnt == 1):# keywords
        #     attr = []
        #     zipf_exponent = 1.2
        #     # cluster centroids for each attr, the closer to the centroid, the higher probability to be selected
        #     # based on the probability, we extrally follows the zipf distribution. for example: '0' with about 0.8, '8' with about 0.001(not accurate because probability depends on distance)
        #     # distribution algorithm: distance_prob * zipf_prob
        #     ranks = np.arange(1, attr_cnt + 1)
        #     zipf_probabilities = 1 / (ranks ** zipf_exponent)
        #     zipf_probabilities /= zipf_probabilities.sum()
        #     for idx, _data in enumerate(data):
        #         # for each v
        #         # print("idx:", idx)
        #         distance = np.linalg.norm(_data - centroids, axis=1)
        #         probabilities = 1 - distance/np.sum(distance)
        #         probabilities = probabilities * zipf_probabilities # dot product
        #         generated_attrs = []
        #         while(True):
        #             for i in range(attr_cnt):
        #                 val = random.random()
        #                 if val <= probabilities[i]:
        #                     generated_attrs.append(i)
        #             if len(generated_attrs) == 0:
        #                 probabilities = np.sqrt(probabilities)
        #             else:
        #                 break
        #         attr.append(generated_attrs)
                
        # elif(attr_cnt > 1 and query_attr_cnt > 1):# multi-attr
        #     dis_sum = np.sum(distance)
        #     probabilities = 1 - distance/dis_sum
        #     selected_element = random.choices(elements, weights=probabilities, k=attr_cnt)
        #     attr[idx] = selected_element

            

        # int range, to match diskann that only one attr supported, range is [a, a]
        else:
            print("err no such attr_cnt:", attr_cnt)
            exit()
        
    else:
        print("err no such distribution:", distribution)
        exit()


    return attr

def write_attr_json(filepath, attr):
    if not ".json" in filepath:
        print("error, json should be stored in .json file, not ", filepath)
    with open(filepath, 'w') as file:
        json.dump(attr, file, indent=4)

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

def load_data(file):
    if("ivecs" in file):
        data = ivecs_read(file)
    elif("fvecs" in file):
        data = fvecs_read(file)
    elif("bvecs" in file):
        data = bvecs_read(file)
    else:
        print ("cannot support such file type:", file)
        exit()
    return data

def check_data_size(datasetfile, queryfile, db_size, query_size):
    dataset_n, dataset_d = get_data_size(datasetfile)
    query_n, query_d = get_data_size(queryfile)

    if not query_d == dataset_d:
        print("error query and dataset dim not match")
        exit()
    if not dataset_n == db_size:
        print("ori db size:", db_size, " new N: ", dataset_n)
    if not query_n == query_size:
        print("ori query size: ", query_size, " new query size: ", query_n)
    
    return db_size, query_size

if __name__ == "__main__":
    # ${dataset} 
    # ${N} 
    # ${query_size} 
    # ${dataset_file} 
    # ${query_file} 
    # ${output_dataset_attr_file} 
    # ${output_query_range_file} 
    # ${method}
    if len(sys.argv) < 9 :
        print("error wrong argument")
        exit()
    else:

        db_size = int(sys.argv[1])
        print("dataset size: ", db_size)
        if not (isinstance(db_size, int) and db_size > 0):
            print("error invalid N, which should be positive integer")
            exit()
        
        dataset_file = sys.argv[2]
        print("dataset file:", dataset_file)
        check_file(dataset_file)

        output_dataset_attr_file = sys.argv[3]
        print("attr file:", output_dataset_attr_file)
        check_dir(output_dataset_attr_file)

        # method_str = sys.argv[4]
        # print("using algo: ", method_str)
        attr_cnt = int(sys.argv[4])
        print("attr count:", attr_cnt)

        attr_range = int(sys.argv[5])
        print("attr range:[0,", attr_range, ")")
        
        distribution = sys.argv[6]
        print("attr distribution:", distribution)

        # query_file = sys.argv[7]
        # print("query file:", query_file)
        # check_file(query_file)

        # output_query_range_file = sys.argv[8]
        # print("query range file:", output_query_range_file)
        # check_dir(output_query_range_file)

        # query_size = int(sys.argv[9])
        # print("query size:", query_size)

        query_attr_cnt = int(sys.argv[7])

        if distribution == "in_dist" or distribution == "out_dist":
            train_file = sys.argv[8]
            print("train file:", train_file)

            train_size = int(sys.argv[9])
            print("train size:", train_size)

            centroid_file = sys.argv[10]
            print("centroid file:", centroid_file)
        
        query_label = int(sys.argv[11]) # not only for query label but for selectivity
        # 1: 0.01, 2: 0.02, 3: 0.03 ....., 10: 0.1, 11: 0.2, ..., 19: 1
        
        if distribution == "real":
            real_label = True
            real_label_file = sys.argv[13]
            print("real label file:", real_label_file)

            # copy label file to output_dataset_attr_file
            os.system("cp " + real_label_file + " " + output_dataset_attr_file)
            print("copy {} to {}".format(real_label_file, output_dataset_attr_file))
            exit(0)

    # directionary check
    # db_size, query_size = check_data_size(dataset_file, query_file, db_size, query_size)
    if distribution == "in_dist" or distribution == "out_dist":
        data = load_data(dataset_file)
        assert(data.shape[0] == db_size)
        train = load_data(train_file)
        if not train.shape[0] == train_size:
            print("file data size:", train.shape[0], " not match input size:", train_size)
            train_idx = np.random.choice(data.shape[0], train_size, replace=False)
            train = data[train_idx]
        print("train shape:", train.shape)
        #generate attributions
        attr = genearte_attr(db_size, attr_cnt, attr_range, distribution, query_attr_cnt, query_label=query_label, data=data, train=train, centroid_file=centroid_file)
    else:
        attr = genearte_attr(db_size, attr_cnt, attr_range, distribution, query_attr_cnt, query_label=query_label)

    #write attribution and query range to file
    # write_attr(output_dataset_attr_file, attr)
    # write_query_range(output_query_range_file, queryattr)

    write_attr_json(output_dataset_attr_file, attr)
    #debug
    for i in range(5):
        print("data attr ", i, " attr=", attr[i])
    
