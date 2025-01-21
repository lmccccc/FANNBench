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
    ranks = np.arange(1, num_attrs + 1)

    # Calculate probabilities using Zipf's law: p(k) ∝ 1 / k^s
    probabilities = 1 / (ranks ** zipf_exponent)

    # Normalize the probabilities to sum to 1
    probabilities /= probabilities.sum()
    generated_attrs = []
    while(len(generated_attrs) == 0):
        for i in range(num_attrs):
            val = random.random()
            if val <= probabilities[i]:
                generated_attrs.append(i)
    return generated_attrs

# input: database vector size, query size, attribution range(like [0,1)), query count(like {[0,0.2), [0.3, 0.5))})
def genearte_attr(db_size, attr_cnt, attr_range, distribution, query_attr_cnt, data=None, train=None, centroid_file=None):
    if(attr_range == -1):
        attr_range = 2**31 - 1
    #generate attribution and query range
    if(distribution == "random"):
        print("generating for dist:", distribution)
        if(attr_cnt == 1):
            #attr format: random integer from 0 to 
            attr = np.random.randint(0, attr_range, db_size, dtype='int32').reshape(db_size).tolist()

        elif(attr_cnt > 1 and query_attr_cnt == 1):# keywords
            attr = []
            zipf_exponent = 1.2
            for data_ind in range(db_size):
                attrs = generate_zipf_attrs(attr_cnt, zipf_exponent)
                attr.append(attrs)
        elif(attr_cnt > 1 and query_attr_cnt > 1):# multi-attr
            attr = np.random.randint(0, attr_range, (db_size, attr_cnt), dtype='int32').tolist()

            
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

        if(attr_cnt == 1): # range query attrs
            elements = [i for i in range(centroid_size)]
            attr = np.zeros((data.shape[0]), dtype='int32').tolist()
            segment_size = math.ceil(attr_range / centroid_size)
            centroid_vals = [segment_size * i + round(segment_size / 2) for i in range(centroid_size)]
            print("centroid_vals:", centroid_vals)
            print("centroid_size:", len(centroid_vals))
            for idx, _data in enumerate(data):
                distance = np.linalg.norm(_data - centroids, axis=1)
                dis_sum = np.sum(distance)
                probabilities = 1 - distance/dis_sum
                selected_element = random.choices(elements, weights=probabilities, k=1)[0]
                
                if attr_range <=128:
                    attr[idx] = selected_element
                else:
                    # generate random int by normal distribution (selected_element[0], attr_range/centroid_size)
                    mean = segment_size * selected_element + round(segment_size / 2)
                    var = math.sqrt(attr_range/centroid_size)
                    while True:
                        attr[idx] = int(np.random.normal(mean, var))
                        if attr[idx] >= 0 and attr[idx] < attr_range:
                            break
                 
        

        elif(attr_cnt > 1 and query_attr_cnt == 1):# keywords
            attr = []
            zipf_exponent = 1.2
            # cluster centroids for each attr, the closer to the centroid, the higher probability to be selected
            # based on the probability, we extrally follows the zipf distribution. for example: '0' with about 0.8, '8' with about 0.001(not accurate because probability depends on distance)
            # distribution algorithm: distance_prob * zipf_prob
            ranks = np.arange(1, attr_cnt + 1)
            zipf_probabilities = 1 / (ranks ** zipf_exponent)
            zipf_probabilities /= zipf_probabilities.sum()
            for idx, _data in enumerate(data):
                # for each v
                # print("idx:", idx)
                distance = np.linalg.norm(_data - centroids, axis=1)
                probabilities = 1 - distance/np.sum(distance)
                probabilities = probabilities * zipf_probabilities # dot product
                generated_attrs = []
                while(True):
                    for i in range(attr_cnt):
                        val = random.random()
                        if val <= probabilities[i]:
                            generated_attrs.append(i)
                    if len(generated_attrs) == 0:
                        probabilities = np.sqrt(probabilities)
                    else:
                        break
                attr.append(generated_attrs)
                
        elif(attr_cnt > 1 and query_attr_cnt > 1):# multi-attr
            dis_sum = np.sum(distance)
            probabilities = 1 - distance/dis_sum
            selected_element = random.choices(elements, weights=probabilities, k=attr_cnt)
            attr[idx] = selected_element

            

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
    if len(sys.argv) < 8 :
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

    # directionary check
    # db_size, query_size = check_data_size(dataset_file, query_file, db_size, query_size)
    if distribution == "in_dist" or distribution == "out_dist":
        data = load_data(dataset_file)
        assert(data.shape[0] == db_size)
        train = load_data(train_file)
        if not train.shape[0] == train_size:
            print("file data size:", train.shape[0], " not match input size:", train_size)
            data = np.random.choice(data, train_size, replace=False)
        print("train shape:", train.shape)
        #generate attributions
        attr = genearte_attr(db_size, attr_cnt, attr_range, distribution, query_attr_cnt, data, train, centroid_file)
    else:
        attr = genearte_attr(db_size, attr_cnt, attr_range, distribution, query_attr_cnt)

    #write attribution and query range to file
    # write_attr(output_dataset_attr_file, attr)
    # write_query_range(output_query_range_file, queryattr)

    write_attr_json(output_dataset_attr_file, attr)
    #debug
    for i in range(5):
        print("data attr ", i, " attr=", attr[i])
    
