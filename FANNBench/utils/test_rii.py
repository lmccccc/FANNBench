import rii
import nanopq
import sys
import numpy as np
import math
import time
import os
import pickle
from defination import check_dir, check_file, ivecs_read, fvecs_read, bvecs_read, read_attr, read_file

if __name__ == "__main__":
    # ${dataset} 
    # ${N} 
    # ${query_size} 
    # ${dataset_file} 
    # ${query_file} 
    # ${output_dataset_attr_file} 
    # ${output_query_range_file} 
    # ${method}
    if not len(sys.argv) == 11 :
        print("error wrong argument")
        exit()
    else:
        N = int(sys.argv[1])
        print("N:", N)
        
        dataset_file = sys.argv[2]
        print("dataset file:", dataset_file)
        check_file(dataset_file)

        query_file = sys.argv[3]
        print("query file:", query_file)
        check_file(query_file)

        train_file = sys.argv[4]
        print("train file:", train_file)
        check_file(train_file)

        dataset_attr_file = sys.argv[5]
        print("attr file:", dataset_attr_file)
        check_file(dataset_attr_file)

        query_range_file = sys.argv[6]
        print("query range file:", query_range_file)
        check_file(query_range_file)

        groundtruth_file = sys.argv[7]
        print("ground truth file:", groundtruth_file)
        check_file(groundtruth_file)

        M = int(sys.argv[8])
        print("partition size M:", M)

        K = int(sys.argv[9])
        print("K:", K)

        index_file = sys.argv[10]
        print("index file location:", index_file)


    # N, Nt, D = 10000, 1000, 128
    # X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be searched
    # Xt = np.random.random((Nt, D)).astype(np.float32)  # 1,000 128-dim vectors for training
    # q = np.random.random((D,)).astype(np.float32)  # a 128-dim vector
    dataset = read_file(dataset_file)
    N, d = dataset.shape
    print("get dataset size:", N, " d:", d)

    query = read_file(query_file)
    Nq, _d = query.shape
    assert(_d == d)
    print("get query szie:", Nq)

    train = read_file(train_file)
    Nt, _d = train.shape
    assert(_d == d)
    print("get train szie:", Nt)

    attr = read_attr(dataset_attr_file)
    _N = len(attr)
    print("_N:", _N, " N:", N)
    assert(_N == N)

    qrange = read_attr(query_range_file)
    _Nq = len(qrange)
    _res = _Nq / Nq
    print("get range dim:", _res)
    assert(isinstance(_res, int) or (isinstance(_res, float) and _res.is_integer()))

    gt = read_attr(groundtruth_file)
    Ngt = len(gt)
    _res = Ngt / Nq
    print("groundtruth top K: ", _res)
    print("ngt=", Ngt, " nq=", Nq, " k=", K)
    assert(Ngt == Nq * K)

    if os.path.isfile(index_file):
        print("load index from ", index_file)
        with open (index_file, 'rb') as fp:
            e = pickle.load(fp)
    else:
        # Prepare a PQ/OPQ codec with M=32 sub spaces
        t0 = time.time()
        codec = nanopq.PQ(M=M).fit(vecs=train)  # Trained using Xt
        t1 = time.time()
        t_train = t1 - t0
        # Instantiate a Rii class with the codec
        e = rii.Rii(fine_quantizer=codec)
        # Add vectors
        t0 = time.time()
        e.add_configure(vecs=dataset)
        t1 = time.time()
        t_add = t1 - t0
        print("train time: ", t_train, " add time: ", t_add, " construction time: ", t_train + t_add)
        print("index save to ", index_file)
        with open(index_file, 'wb') as fp:
            pickle.dump(e, fp)

    # Search
    t_total = 0
    ids = np.zeros(Nq * K, dtype='int')
    dists = np.zeros(Nq * K, dtype='float')
    for ind, q in enumerate(query):
        condition = (attr>=qrange[ind*2]) & (attr<= qrange[ind*2+1])
        subset = np.where(condition)[0]
        t0 = time.time()
        _ids, _dists = e.query(q=q, topk=K, target_ids=subset)
        t1 = time.time()
        t_total += t1 - t0
        dists[ind*K: ind*K + K] = _dists
        ids[ind*K: ind*K + K] = _ids

    #stat
    
    total_size = Nq * K
    positive_size = 0
    hist = np.array([0 for _ in range(11)], dtype='float')
    for qind in range(Nq):
        q_cnt = 0
        for top in range(K):
            for gtind in range(K):
                if(ids[qind * K + top] == gt[qind * K + gtind]):
                    positive_size += 1
                    q_cnt += 1
        q_recall = q_cnt * 1.0 / K
        q_bucket = math.floor(q_recall * 10)
        hist[q_bucket] += 1
    recall = positive_size * 1.0 / total_size
    print("rii get ", positive_size, " postive res from ", total_size, " results, recall@", K, ":", recall)
    hist = hist / Nq
    print("recall dist(10\% per bucket): ", hist)
    print("qps: ", Nq / t_total)
    # print(ids, dists)  # e.g., [7484 8173 1556] [15.06257439 15.38533878 16.16935158]