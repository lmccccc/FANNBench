import os

import hnswlib
import numpy as np
from alg_base import BaseANN


class HnswLib(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)
        # self.ef=query_param['ef']
        method_param_copy = method_param.copy()
        if "ef" in method_param:
            del method_param_copy["ef"]
        self.name = "hnswlib (%s)" % (method_param_copy)

    def fit(self, X, scalars):
        self.scalars = scalars
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        self.p.init_index(
            max_elements=len(X),
            ef_construction=self.method_param["efConstruction"],
            M=self.method_param["M"],
        )
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def set_query_arguments(self, arg_dict: dict):
        self.p.set_ef(arg_dict["ef"])

    def knn_query(self, q, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        I, D = self.p.knn_query(q, n)
        return I.ravel()

    def hybrid_query(self, q, interval, n):
        low, high = interval
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        I, D = self.p.knn_query(q, self.p.ef)  # p.ef>k?
        I = I.ravel()
        scalars = self.scalars[I.ravel()]
        qualified_knn = I[(scalars >= low) & (scalars <= high)]
        return qualified_knn[:n]

    def freeIndex(self):
        del self.p
        del self.scalars

    def saveIndex(self, location):
        self.p.save_index(location)

    def loadIndex(self, X, location):
        # self.freeIndex()
        n_base, dim = X.shape
        self.p = hnswlib.Index(space=self.metric, dim=dim)
        self.p.load_index(location, n_base)
