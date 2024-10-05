import numpy as np
import sys
import json


dataset_attr_file = "../ACORN/testing_data/sift_attr.ivecs"
attr_output_file  = "../ACORN/testing_data/sift_attr.json"

qrange_file = "../ACORN/testing_data/sift_qrange.ivecs"
qrange_output_file  = "../ACORN/testing_data/sift_qrange.json"

groundtruth_file  = "../ACORN/testing_data/sift_gt.txt"   #stored like ivecs, but long int in faiss
gtoutput_file     = "../ACORN/testing_data/sift_gt_10.json"

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def livecs_read(fname):
    a = np.fromfile(fname, dtype='int64')
    if sys.byteorder == 'big':
        a.byteswap(inplace=True)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

#attr
data = ivecs_read(dataset_attr_file)
data = data.reshape(-1).tolist()
with open(attr_output_file, 'w') as file:
    json.dump(data, file)

#qrange
data = ivecs_read(qrange_file)
data = data.reshape(-1).tolist()
with open(qrange_output_file, 'w') as file:
    json.dump(data, file)

# ground truth
data = livecs_read(groundtruth_file)
data = data.reshape(-1).tolist()
with open(gtoutput_file, 'w') as file:
    json.dump(data, file)
