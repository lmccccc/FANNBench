from defination import fvecs_write, write_attr_json
import numpy as np
from scipy.sparse import csr_matrix
import json
import os

def read_i8bin(filename):
    with open(filename) as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        print("Header:", header)
        dim = header[1]

        d = np.fromfile(f, dtype=np.int8)
        d = d.reshape(-1, dim)
    d = d.astype(np.float32)
    return d, dim

def read_sparse_matrix_fields(fname):
    """ read the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes                                      # shape of matrix shape[0] shape[1] nnz
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)       
        assert nnz == indptr[-1]                                
        indices = np.fromfile(f, dtype='int32', count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype='float32', count=nnz)
        return data, indices, indptr, ncol

def read_sparse_matrix(fname):
    data, indices, indptr, ncol = read_sparse_matrix_fields(fname)
    return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncol))

def csr_matrix_to_list(matrix):
    data_list = [[] for _ in range(matrix.shape[0])]
    coo = matrix.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        data_list[i].append(int(j))
    return data_list

size = 10000000 # 10M
train_size = 1000000 # 1M
qsize = 10000 # 10k

filename = "/mnt/data/mocheng/dataset/spacev/base.1B.i8bin"
query_filename = "/mnt/data/mocheng/dataset/spacev/query.30K.i8bin"
data, dim = read_i8bin(filename)
query_data, dim2 = read_i8bin(query_filename)
assert dim == dim2
print("data shape:", data.shape)
print("query data shape:", query_data.shape)

np.random.seed(0)
sampled_indices = np.random.choice(len(data), size=size, replace=False)
data = data[sampled_indices]
train = data[:train_size]
query = query_data[:qsize]

output_root = "/mnt/data/mocheng/dataset/spacev10m/"
output_file = "/mnt/data/mocheng/dataset/spacev10m/base10M.fvecs"
output_query_file = "/mnt/data/mocheng/dataset/spacev10m/query10k.fvecs"
output_train_file = "/mnt/data/mocheng/dataset/spacev10m/train.fvecs"

if not os.path.exists(output_root):
    os.makedirs(output_root)

print("data shape: ", data.shape)
fvecs_write(output_file, data)
print("train shape: ", train.shape)
fvecs_write(output_train_file, train)
print("query shape: ", query.shape)
fvecs_write(output_query_file, query)

print("done")