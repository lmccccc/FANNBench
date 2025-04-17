from defination import fvecs_write, write_attr_json
import numpy as np
from scipy.sparse import csr_matrix
import json


def read_u8bin(filename, dim):
    with open(filename) as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        print("Header:", header)

        d = np.fromfile(f, dtype=np.uint8)
        d = d.reshape(-1, dim)
    d = d.astype(np.float32)
    return d

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


filename = "/path/yfcc100M/base.10M.u8bin"          # need modify
metadata_filename = "/path/yfcc100M/base.metadata.10M.spmat"        # need modify
query_filename = "/path/yfcc100M/query.public.100K.u8bin"        # need modify
query_metaedata_filename = "/path/yfcc100M/query.metadata.public.100K.spmat"        # need modify
dim = 192

size = 10000000
trainsize = 100000
querysize = 10000

data = read_u8bin(filename, dim)
print("data shape:", data.shape)
np.random.seed(0)
sampled_indices = np.random.choice(len(data), size=size, replace=False)
data = data[sampled_indices]
train_data = data[:trainsize]

query_data = read_u8bin(query_filename, dim)
query_data = query_data[:querysize]

metadata = read_sparse_matrix(metadata_filename)
query_metadata = read_sparse_matrix(query_metaedata_filename)
query_metadata = query_metadata[:querysize]

print("data shape:", data.shape)
print("query data shape:", query_data.shape)


output_file = "/path/yfcc10m/base10M.fvecs"
output_train_file = "/path/yfcc10m/train.fvecs"
output_query_file = "/path/yfcc10m/query10k.fvecs"
output_attr_file = "/path/yfcc10m/attr10M.json"
output_query_attr_file = "/path/yfcc10m/query_attr10k.json"
fvecs_write(output_file, data)
fvecs_write(output_train_file, train_data)
fvecs_write(output_query_file, query_data)

list_metadata = csr_matrix_to_list(metadata)
list_query_metadata = csr_matrix_to_list(query_metadata)
print("list_metadata shape:", len(list_metadata))
write_attr_json(output_attr_file, list_metadata)
write_attr_json(output_query_attr_file, list_query_metadata)

print("done")