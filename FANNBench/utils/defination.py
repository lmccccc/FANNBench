# attribution type, don't change its sequence
import numpy as np
import sys
import os
import json

# method_list = {
#     0 : "random range",
#     1 : "random keyword",
#     2 : "random int range"
# }


###              varieties definiation         ###

# attr_size  = 1000000         # 1M
# query_size = 10000           # 10k
# data_type  = "sift1M"

# method      = None # "random range" or "random keyword" "random int range"
# range_bound = None
# range_size  = None

# def set_range(method_id):
#     method = method_id
#     global range_bound
#     global range_size
#     if(method == method_list[0]):
#         range_bound  = 1             # [0, 1)
#         range_size   = 1             # range count

#     elif(method == method_list[1]): # keyword
#         range_bound    = 12          # range, integer from 1 to key_range
#         range_size   = 1             # range count

#     elif(method == method_list[2]): # range
#         range_bound    = 12          # keyword, integer from 1 to key_range
#         range_size   = 1             # range count
#     else:
#         print("error no such method")
#         exit()
#     return range_bound, range_size


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    if sys.byteorder == 'big':
        m1.byteswap(inplace=True)
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    if sys.byteorder == 'big':
        a.byteswap(inplace=True)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def bvecs_read(fname):
    with open(fname, 'rb') as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]
        vectors = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, d + 4)
        vectors = vectors[:, 4:]
    return vectors


def check_dir(file_dir):
    directory = os.path.dirname(file_dir)
    if(not os.path.isdir(directory)):
        print("error no such directionary: ", directory)
        exit()

def check_file(f):
    if(not os.path.isfile(f)):
        print("error no such directionary: ", f)
        exit()

def read_attr(fname):
    with open(fname, 'r') as file:
        data = json.load(file)
    assert(isinstance(data, list))
    return np.array(data, dtype="int64")

def read_file(file):
    if("ivecs" in file):
        data = ivecs_read(file)
    elif("fvecs" in file):
        data = fvecs_read(file)
    elif("bvecs" in file):
        data = bvecs_read(file)
    else:
        print ("cannot support such file type:", file)
        exit()
    return data