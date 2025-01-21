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

filename = "/mnt/data/mocheng/dataset/yfcc100M/base.10M.u8bin"
metadata_filename = "/mnt/data/mocheng/dataset/yfcc100M/base.metadata.10M.spmat"
query_filename = "/mnt/data/mocheng/dataset/yfcc100M/query.public.100K.u8bin"
query_metaedata_filename = "/mnt/data/mocheng/dataset/yfcc100M/query.metadata.public.100K.spmat"
dim = 192
data = read_u8bin(filename, dim)
query_data = read_u8bin(query_filename, dim)
metadata = read_sparse_matrix(metadata_filename)
query_metadata = read_sparse_matrix(query_metaedata_filename)

print("data shape:", data.shape)
print("query data shape:", query_data.shape)


output_file = "/mnt/data/mocheng/dataset/yfcc/base10M.fvecs"
output_query_file = "/mnt/data/mocheng/dataset/yfcc/query100k.fvecs"
output_query_file2 = "/mnt/data/mocheng/dataset/yfcc/query10k.fvecs"
output_attr_file = "/mnt/data/mocheng/dataset/yfcc/attr10M.json"
output_query_attr_file = "/mnt/data/mocheng/dataset/yfcc/query_attr100k.json"
output_query_attr_file2 = "/mnt/data/mocheng/dataset/yfcc/query_attr10k.json"
fvecs_write(output_file, data)
fvecs_write(output_query_file, query_data)
fvecs_write(output_query_file2, query_data[:10000])

list_metadata = csr_matrix_to_list(metadata)
list_query_metadata = csr_matrix_to_list(query_metadata)
print("list_metadata shape:", len(list_metadata))
write_attr_json(output_attr_file, list_metadata)
write_attr_json(output_query_attr_file, list_query_metadata)
write_attr_json(output_query_attr_file2, list_query_metadata[:10000])

print("done")