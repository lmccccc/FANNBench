import argparse
import os
import time
from typing import List

import h5py
import numpy as np
import pandas as pd
from alg_hannlib import HannLib
from alg_hnswlib import HnswLib


def read_hdf5_dataset(filepath, keys: List[str]):
    with h5py.File(filepath, "r") as f:
        ret = []
        for k in keys:
            ret.append(f[k][:])
    return ret


def build_index(name: str, params, base, base_scalars, index_save_path):
    index = None
    if name == "HNSW":
        index = HnswLib(metric="euclidean", method_param=params)
    else:
        index = HannLib(metric="euclidean", method_param=params)
    # else:
    #     index = FaissIvfPq(metric="euclidean", method_param=params)

    if os.path.exists(index_save_path):
        print(f"Reading index from {index_save_path} ...")
        start, end = 0, 0
        index.loadIndex(base, index_save_path)
        index.scalars = base
    else:
        print(f"Building index: {index.name}...")
        start = time.time()
        index.fit(base, base_scalars)
        end = time.time()
        index.saveIndex(index_save_path)
        if name == "MBV-HNSW":
            index.set_query_arguments(ef=1, al=params["al"])

        print(f"Index built: {index.name}, duration: {end-start}.")
        with open(index_save_path + ".time", "w") as f:
            f.write(f"{end - start}")

    return index, end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the hdf5 data file"
    )
    parser.add_argument(
        "--use_mbv_hnsw", type=bool, default=False, help="Whether to use MBV-HNSW index"
    )
    parser.add_argument("--M", type=int, default=16, help="Number of graph connections")
    parser.add_argument(
        "--efConstruction", type=int, default=500, help="Parameter for HNSW index"
    )
    parser.add_argument("--B", type=int, default=8, help="Number of buckets")
    parser.add_argument(
        "--save_path", type=str, required=True, help="Directory to save the index"
    )
    args = parser.parse_args()

    if args.use_mbv_hnsw:
        name = "MBV-HNSW"
        params = {
            "num_slots": args.B,
            "M": args.M,
            "efConstruction": args.efConstruction,
            "al": 16,
        }
    else:
        name = "HNSW"
        params = {"M": args.M, "efConstruction": args.efConstruction}

    (
        base,
        base_scalars,
        test,
        test_ranges,
        test_knn,
        test_hybrid_knn,
    ) = read_hdf5_dataset(
        args.data_path,
        ["base", "base_scalars", "test", "test_ranges", "test_knn", "test_hybrid_knn"],
    )

    build_index(name, params, base, base_scalars, args.save_path)
