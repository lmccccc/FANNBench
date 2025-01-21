import faiss
import numpy as np


def build_brute_force_index(base: np.ndarray, space="l2"):
    dim = base.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(base)
    return index


def brute_force_knn(base: np.ndarray, query: np.ndarray, k: int):
    index = build_brute_force_index(base)
    return index.search(query, k)
