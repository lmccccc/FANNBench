from multiprocessing.pool import Pool, ThreadPool
from typing import List

import h5py
# import numba as nb
import numpy as np

# from numba import njit


class ParallelMapper:
    def __init__(self, num_threads, use_process=True) -> None:
        self.num_threads = num_threads
        if use_process:
            self.pool = Pool(num_threads)
        else:
            self.pool = ThreadPool(num_threads)

    def map(self, func, container):
        results = self.pool.map(func, container)
        return results


# @njit
def compute_recall(preds, targets, k: int, m: int):
    """compute recall_k@m"""
    total_correct = 0
    total = 0
    nq = len(targets)
    for i in range(0, nq):
        gt_i = targets[i, :k].astype(np.int64)
        got_i = preds[i, :m].astype(np.int64)
        correct = len(set(gt_i) & set(got_i))
        count = len(set(gt_i))
        # if correct != count:
        #     print(f"Exptected: {gt_i}, got: {got_i}")
        total_correct += correct
        total += count
    return total_correct / total


# @njit
def compute_rowwise_recall(preds, targets, k: int, m: int):
    """compute recall_k@m"""
    nq = len(targets)
    recalls = np.zeros(len(preds), dtype=nb.float32)
    for i in range(0, nq):
        gt_i = targets[i, :k].astype(np.int64)
        got_i = preds[i, :m].astype(np.int64)
        correct = len(set(gt_i) & set(got_i))
        recalls[i] = correct / k
    return recalls


def read_hdf5_dataset(filepath, keys: List[str]):
    with h5py.File(filepath, "r") as f:
        ret = []
        for k in keys:
            ret.append(f[k][:])
    return ret


def write_hdf5_dataset(data_dict: dict, output_path):
    with h5py.File(output_path, "w") as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v, compression="lzf")


def euc_normalize(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """normalize vectors in `x` to the range of [0,1]

    Args:
        x (np.ndarray): [description]
        min_value (float): [description]
        max_value (float): [description]

    Returns:
        np.ndarray: [description]
    """
    return (x - min_value) / (max_value - min_value)


def cos_normalize(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """normalization function for cosine space datasets
    Args:
        x (np.ndarray): the vectors
        min_value (float): this value is fully ignored
        max_value (float): this value is fully ignored
    Returns:
        np.ndarray: [description]
    """
    norm = np.linalg.norm(x, axis=1)
    norm.resize((len(norm), 1))
    ret = x / norm
    # set elements of all-zero vectors as a large-enough number (here 100)
    # thus they cannot become KNN of any given queries
    ret[np.isnan(ret)] = 100
    return ret


def random_keys(n: int):
    """generate n unique keys of dtype [uint64]"""
    result_set = set()

    while len(result_set) < n:
        keys = np.random.randint(0, n * 10, n, dtype="uint64")
        result_set = result_set.union(keys)

    return np.array(list(result_set))[:n]


def compute_slot_ranges(scalars, num_slots):
    step = 1 / num_slots * 100
    percentiles = []
    for i in range(1, num_slots):
        percentiles.append(np.round(step * i))
    slot_ranges = np.zeros((num_slots, 2), dtype="int64")
    values = np.percentile(
        scalars, percentiles
    )  # np.percentile(scalars, p)返回的是scalar中有百分之p的数小于该值
    for i in range(0, num_slots):
        if i == 0:
            slot_ranges[i, 0] = scalars.min()
        else:
            slot_ranges[i, 0] = values[i - 1]

        if i == num_slots - 1:
            slot_ranges[i, 1] = scalars.max()
        else:
            slot_ranges[i, 1] = values[i]
    return slot_ranges
