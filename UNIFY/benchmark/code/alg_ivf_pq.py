import json
import os
from enum import Enum

import hannlib_baseline_avx2 as hb
import numpy as np
from alg_base import BaseANN

hb.set_verbose_level(0)


class IndexFamily:
    kVectorIndex = 0
    # # Not supported now:
    # kTextIndex = 1


class IndexType:
    kHnswlib = 0
    kFaissIvfPq = 2
    kFaissBruteforce = 3


class MetricType:
    kL2Distance = 0
    # # Not supported now:
    # kCosineSimilarity = 1
    # kInnerProduct = 2
    # kCosineDistance = 3


def get_ivfpq_meta(dim,M, nlist, nbits=8):
    # init index meta
    meta = {}

    meta["meta_version"] = 0
    meta["family"] = IndexFamily.kVectorIndex
    meta["type"] = IndexType.kFaissIvfPq

    common_params = {}
    common_params["dim"] = dim
    common_params["metric_type"] = MetricType.kL2Distance
    common_params["is_vector_normed"] = False

    index_params = {}
    index_params["M"] = M
    index_params["nlist"] = nlist
    index_params["nbits"] = nbits

    search_params = {}
    search_params["nprobe"] = nlist

    extra_params = {}
    extra_params["comments"] = "IVFPQ"

    meta["common"] = common_params
    meta["index"] = index_params
    meta["search"] = search_params
    meta["extra"] = extra_params

    return meta


def test_search(hybrid_index, low: int, high: int, query, k: int, search_params=None):
    print(f"========== {hybrid_index} ==========")
    if search_params is not None:
        hybrid_index.set_search_parameters(search_params)
    I, D = hybrid_index.search(low, high, query, 10)
    print(f"---------- pre-filtering ----------")
    print(I)
    print(D)

    print(f"---------- post-filtering ----------")
    I, D = hybrid_index.search(low, high, query, k, False)
    print(I)
    print(D)


def test_index(index_meta, x_vector, x_scalar, search_params=None, num_build_threads=8):
    hybrid_index = hb.HybridIndex(json.dumps(index_meta))
    hybrid_index.num_build_threads = num_build_threads
    hybrid_index.train(x_vector, x_scalar)
    hybrid_index.add(x_vector, x_scalar)

    test_search(hybrid_index, 0, 100, x_vector[0], 10, search_params)
    hb.save_index(str(hybrid_index), hybrid_index)
    
    print("========== Loading index ==========")
    hybrid_index1 = hb.load_index(str(hybrid_index))
    test_search(hybrid_index1, 0, 100, x_vector[0], 10, search_params)

class FaissIvfPq(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        self.name = "faiss-ivf-pq (%s)" % (self.method_param)
        
        self.M = method_param["M"]
        self.nlist = method_param["nlist"]
        self.nbits= method_param["nbits"]
        
        self.p = None


    def fit(self, X, scalars):
        print("X.shape:", X.shape)
        print("scalars.shape:", scalars.shape)
        _, dim = X.shape
        index_meta = get_ivfpq_meta(dim, self.M, self.nlist, self.nbits)
        meta_str = json.dumps(index_meta)
        print(meta_str)
        self.p = hb.HybridIndex(meta_str)
        n_cpu = os.cpu_count()
        self.p.num_build_threads = n_cpu - 4 if n_cpu > 4 else n_cpu
        
        self.p.train(X, scalars)
        self.p.add(X, scalars)

    def set_query_arguments(self, params):
        self.p.set_search_parameters(json.dumps(params))

    def knn_query(self, q, k):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        I, D = self.p.search(q, k)
        return I.ravel()

    def hybrid_query(self, q, interval, k):
        assert len(q.shape) == 1 # q must be a 1D vector
        
        low, high = interval
        I, D = self.p.search(low, high, q, k)
        I = I.ravel()
        return I

    def freeIndex(self):
        del self.p
        
    def saveIndex(self, location):
        hb.save_index(location, self.p)
    
    def loadIndex(self, X, location):
        self.p = hb.load_index(location)

if __name__ == "__main__":
    # generate random data for indexing
    np.random.seed(0)
    d = 128
    nb = 1000000
    nq = 100
    xb = np.random.random((nb, d)).astype("float32")
    xq = np.random.random((nq, d)).astype("float32")

    hb.set_verbose_level(100)
    # create a Faiss index
    index = FaissIvfPq("euclidean", {"nlist": 4096, "M":64, "nbits":8})
    index.fit(xb, np.arange(len(xb)))

    I = index.hybrid_query(xq[0], (0,100), 10)
    print(I)