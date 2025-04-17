import numpy as np
from defination import fvecs_write
import json
import os

root = "/path/dataset/redcaps/"                # need modify
output_folder1 = "/path/dataset/redcaps1m/"    # need modify
output_folder2 = "/path/dataset/redcaps4m/"    # need modify

file = root + "image_embeddings/redcaps-512-angular.npy"                                   
timestame_file = "image_embeddings/redcaps-512-angular_filter-timestamps.npy"

size1 = 1000000
size2 = 4000000
query_size = 10000
train_size = 100000
data_file = "image_embeddings.fvecs"
train_file = "train.fvecs"
attr_file = "timestamp.json"
query_file = "query.fvecs"

if not os.path.exists(output_folder1):
    os.makedirs(output_folder1)
if not os.path.exists(output_folder2):  
    os.makedirs(output_folder2)



all_data = np.load(file)
all_timestamps = np.load(timestame_file)
print("shape of data:", all_data.shape)
print("shape of timestamps:", all_timestamps.shape)

total_size = all_data.shape[0]
timestamp_min = np.min(all_timestamps)
timestamp_max = np.max(all_timestamps)
print("min timestamp:", timestamp_min)
print("max timestamp:", timestamp_max)

# shuffle index
idx = np.arange(0, total_size)
np.random.shuffle(idx)
data = all_data[idx[:size1]]
query_data = all_data[idx[size1:size1+query_size]]
timestamps = all_timestamps[idx[:size1]]
fvecs_write(output_folder1 + data_file, data)
fvecs_write(output_folder1 + train_file, data[:train_size])
fvecs_write(output_folder1 + query_file, query_data)
json.dump(timestamps.tolist(), open(output_folder1 + attr_file, "w"))
print("saved", size1, "data to", output_folder1)

data = all_data[idx[:size2]]
query_data = all_data[idx[size1:size1+query_size]]
timestamps = all_timestamps[idx[:size2]]
fvecs_write(output_folder2 + data_file, data)
fvecs_write(output_folder2 + train_file, data[:train_size])
fvecs_write(output_folder2 + query_file, query_data)
json.dump(timestamps.tolist(), open(output_folder2 + attr_file, "w"))
print("saved", size2, "data to", output_folder2)