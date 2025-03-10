import os

import numpy as np
import utils
from alg_base import BaseANN

import hannlib

# import hannlib


class HannLib(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)
        # self.ef=query_param['ef']
        method_param_copy = method_param.copy()
        if "ef" in method_param:
            del method_param_copy["ef"]
        self.name = "hannlib (%s)" % (method_param_copy)

        self.scalar_min = None
        self.scalar_max = None

    def fit(self, X, scalars, threads):
        # Only l2 is supported currently
        n_base, dim = X.shape
        num_slots = self.method_param["num_slots"]
        slot_ranges = utils.compute_slot_ranges(
            scalars, num_slots
        )  # shape(num_slots,2)

        self.scalar_min = scalars.min()
        self.scalar_max = scalars.max()

        print("init index")
        self.p = hannlib.HybridIndex(space=self.metric, dim=dim)
        self.p.init_index(
            slot_ranges=slot_ranges,
            max_elements=n_base,
            ef_construction=self.method_param["efConstruction"],
            M=self.method_param["M"],
        )
        print(f"Adding {len(X)} vectors, {len(scalars)} scalars.")
        data_labels = np.arange(len(X))
        self.p.set_num_threads(threads)
        print("set threads in py: ", threads)
        self.p.add_items(np.asarray(X), scalars, data_labels)
        self.p.set_num_threads(1)
        print("add items done")

    def set_query_arguments(self, arg_dict: dict):
        self.p.set_ef(arg_dict["ef"])
        self.p.set_al(arg_dict["al"])
        self.p.set_search_strategy(arg_dict["search_strategy"])
        self.p.set_target_recall(arg_dict["target_recall"])
        self.p.set_low_range(arg_dict["low_range"])
        self.p.set_high_range(arg_dict["high_range"])
        if "optimizer_conf_dir" in arg_dict:
            conf_dir = arg_dict["optimizer_conf_dir"]
            if type(conf_dir) == str and conf_dir != "":
                self.p.load_optimizer_conf(conf_dir)

    def knn_query(self, q, n):
        I, D = self.p.knn_query(q, n)
        return I
        # interval = np.array([self.scalar_min, self.scalar_max], dtype="int64")
        # I, D = self.p.hybrid_query(q, interval, n)
        # return I

    def hybrid_query(self, q, interval, n):
        I, D, comps = self.p.hybrid_query(q, interval, n)
        return I, comps

    def freeIndex(self):
        del self.p

    def saveIndex(self, location):
        self.p.save_index(location)

    def loadIndex(self, X, location):
        # self.freeIndex()
        print("hannlib load hybrid index")
        n_base, dim = X.shape
        self.p = hannlib.HybridIndex(space=self.metric, dim=dim)
        self.p.load_index(location, n_base)
