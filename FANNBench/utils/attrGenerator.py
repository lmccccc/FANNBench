import numpy as np
import random
import os
from defination import *
import sys
import json
import time



# input: database vector size, query size, attribution range(like [0,1)), query count(like {[0,0.2), [0.3, 0.5))})
def genearte_attr(method, db_size, attr_range, distribution, data=None, train=None, centroid_file=None):
    #generate attribution and query range
    if(distribution == "random"):
        print("generating for dist:", distribution)
        if(method == "range"):# range
            #attr format: random integer from 0 to 300
            attr = np.random.uniform(low=0, high=attr_range, size=(db_size, 1))


        elif(method == "keyword"):# keyword
            #attr format: random integer from 0 to 30
            attr = np.random.randint(0, attr_range, db_size, dtype='int32').reshape(db_size, -1)

        # int range, to match diskann that only one label supported, range is [a, a]
        else:
            print("err no such method:", method)
            exit()
    elif(distribution == "in_dist" or distribution == "out_dist"):
        from sklearn.cluster import KMeans
        assert train is not None
        assert data is not None
        assert centroid_file is not None

        
        if os.path.exists(centroid_file):
            print("load centroids from file:", centroid_file)
            centroids = np.loadtxt(centroid_file)
        else:
            print("generating for dist:", distribution)
            t0 = time.time()
            kmeans = KMeans(n_clusters=attr_range, random_state=42)
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


        if(method == "keyword"):# keyword
            attr = np.zeros((data.shape[0]), dtype='int32')
            for idx, _data in enumerate(data):
                distance = np.linalg.norm(_data - centroids, axis=1)
                min_index = np.where(distance == np.min(distance))[0]
                attr[idx] = min_index[0]
                # print("data ", idx, " attr:", attr[idx])

        # int range, to match diskann that only one label supported, range is [a, a]
        else:
            print("err no such method:", method)
            exit()
        
    else:
        print("err no such distribution:", distribution)
        exit()


    return attr

def write_attr_json(filepath, attr):
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

        attr_size = int(sys.argv[1])
        print("dataset size: ", attr_size)
        if not (isinstance(attr_size, int) and attr_size > 0):
            print("error invalid N, which should be positive integer")
            exit()
        
        dataset_file = sys.argv[2]
        print("dataset file:", dataset_file)
        check_file(dataset_file)

        output_dataset_attr_file = sys.argv[3]
        print("attr file:", output_dataset_attr_file)
        check_dir(output_dataset_attr_file)

        method_str = sys.argv[4]
        print("using algo: ", method_str)

        range_bound = int(sys.argv[5])
        print("attr range:[0,", range_bound, ")")
        
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

        if distribution == "in_dist" or distribution == "out_dist":
            train_file = sys.argv[7]
            print("train file:", train_file)

            train_size = int(sys.argv[8])
            print("train size:", train_size)

            centroid_file = sys.argv[9]
            print("centroid file:", centroid_file)

    # directionary check
    # attr_size, query_size = check_data_size(dataset_file, query_file, attr_size, query_size)
    if distribution == "in_dist" or distribution == "out_dist":
        data = load_data(dataset_file)
        assert(data.shape[0] == attr_size)
        train = load_data(train_file)
        if not train.shape[0] == train_size:
            print("file data size:", train.shape[0], " not match input size:", train_size)
            data = np.random.choice(data, train_size, replace=False)
        print("train shape:", train.shape)
        #generate attributions
        attr = genearte_attr(method_str, attr_size, range_bound, distribution, data, train, centroid_file)
    else:
        attr = genearte_attr(method_str, attr_size, range_bound, distribution)

    #write attribution and query range to file
    # write_attr(output_dataset_attr_file, attr)
    # write_query_range(output_query_range_file, queryattr)

    write_attr_json(output_dataset_attr_file, attr)
    #debug
    for i in range(5):
        print("data attr ", i, " attr=", attr[i])