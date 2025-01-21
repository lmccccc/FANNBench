# attribution type, don't change its sequence
import numpy as np
import sys
import os
import json


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
        # d = np.frombuffer(f.read(4), dtype=np.int32)[0]
        vectors = np.frombuffer(f.read(), dtype=np.uint8)
        d = np.frombuffer(vectors[:4], dtype=np.int32)[0]
        vectors = vectors.reshape(-1, d + 4)
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

def read_keywords(fname):
    with open(fname, 'r') as file:
        data = json.load(file)
    assert(isinstance(data, list))
    return data


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


def write_attr_json(filepath, attr):
    if not ".json" in filepath:
        print("error, json should be stored in .json file, not ", filepath)
    with open(filepath, 'w') as file:
        json.dump(attr, file, indent=4)