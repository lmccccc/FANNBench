from defination import fvecs_read, fvecs_write, bvecs_read
import numpy as np

# def bvecs_read(fname, size):
#     with open(fname, 'rb') as f:
#         d = np.frombuffer(f.read(4), dtype=np.int32)[0]
#         row_dim = d + 4
#         vectors = np.frombuffer(f.read(row_dim * size), dtype=np.uint8).reshape(-1, d + 4)
#         vectors = vectors[:, 4:]
#     return vectors

def bvecs2fvecs(bvecs_file):
    bvecs = bvecs_read(bvecs_file)


    fvecs = bvecs.astype('float32')
    return fvecs

root = "/path/dataset/sift1b/"         # need modify
output_root = "/path/dataset/sift10m/" # need modify
bvecs_file = root + "bigann_base.bvecs"
query_file = root + "bigann_query.bvecs"
fvecs_file = output_root + "sift10m.fvecs"
train_fvecs_file = output_root + "sift10m_train.fvecs"
query_fvecs_file = output_root + "sift10m_query.fvecs"

size = 10000000 # 10M
train_size = 1000000 # 1M
qsize = 10000 # 10k

data = bvecs2fvecs(bvecs_file)
print("load data shape: ", data.shape)
# np.random.shuffle(fvecs)

if size > len(data):
    raise ValueError("Sample size cannot be larger than the dataset size.")

# Randomly sample indices
np.random.seed(0)
sampled_indices = np.random.choice(len(data), size=size, replace=False)

data = data[sampled_indices]
print("data shape: ", data.shape)
fvecs_write(fvecs_file, data)

# Select the sampled vectors
train = data[:train_size]
print("train shape: ", train.shape)
fvecs_write(train_fvecs_file, train)


query_fvecs = bvecs2fvecs(query_file)
query_fvecs = query_fvecs[:qsize]
fvecs_write(query_fvecs_file, query_fvecs)
print("query shape: ", query_fvecs.shape)
print("Done!")

