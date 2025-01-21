from ctypes import util
from multiprocessing.pool import ThreadPool

import faiss_utils
import numpy as np
import utils

np.random.seed(100)

raw_data_root = "/data/home/petrizhang/data/vector/raw"
data_root = "/data/home/petrizhang/data/hybrid_anns"


def run(dataset, filepath):

    train, test, test_knn = utils.read_hdf5_dataset(
        filepath, ["train", "test", "neighbors"]
    )

    # train = train[:11830]

    nq = 1000
    test = test[:nq]
    test_knn = test_knn[:nq]

    if "angular" in dataset:
        train = utils.cos_normalize(train, None, None)
        test = utils.cos_normalize(test, None, None)

    n_train, dim = train.shape
    n_test = test.shape[0]

    scalar_min, scalar_max = 0, n_train
    base_scalars = np.arange(scalar_min, scalar_max, dtype="int64")

    def generate_test_ranges(n, min_value, max_value, min_size, max_size):
        start = np.random.randint(min_value, max_value, n)
        size = np.random.randint(min_size, max_size, n)
        ranges = np.ones([n, 2], dtype="int64")
        ranges[:, 0] = start
        ranges[:, 1] = start + size
        return ranges

    range_size = scalar_max - scalar_min

    test_ranges = generate_test_ranges(
        n_test, scalar_min, scalar_max, np.ceil(range_size * 0.05), range_size
    )

    # generate ground truth

    test_hybrid_knn = np.zeros([n_test, 100], dtype="uint64")
    # test_hybrid_distances = np.zeros([n_test, 100], dtype="float32")

    def bf_hybrid_search(threadId, train, q, query_range):
        print(f"Thread {threadId}, searching range {query_range}...")
        low, high = query_range
        mask = (base_scalars >= low) & (base_scalars <= high)
        id_map = np.arange(0, len(train))[mask]
        base = train[mask]
        D, I = faiss_utils.brute_force_knn(base, q.reshape(1, len(q)), 100)
        print(f"Thread {threadId}, searching done.")
        return D, id_map[I]

    pool = ThreadPool(10)
    results = pool.map(
        lambda q: bf_hybrid_search(q[0], train, q[1], q[2]),
        zip(range(n_test), test, test_ranges),
    )

    i = 0
    for D, I in results:
        test_hybrid_knn[i, :] = I
        # test_hybrid_distances[i, :] = D
        i += 1


    output_path = f"{data_root}/{dataset}_with_scalar.hdf5"
    utils.write_hdf5_dataset(
        {
            "base": train,               # dataset
            "base_scalars": base_scalars, # attr
            "test": test,                  # query
            "test_ranges": test_ranges,     # query range
            "test_knn": test_knn,               # ?
            "test_hybrid_knn": test_hybrid_knn, # ?
        },
        output_path,
    )


dataset_all = [
    # "sift-128-euclidean",
    # "fashion-mnist-784-euclidean",
    # "glove-100-angular",
    # "nytimes-256-angular",
]

for dataset in dataset_all:
    filepath = f"{raw_data_root}/{dataset}.hdf5"
    print(filepath)
    run(dataset, filepath)
