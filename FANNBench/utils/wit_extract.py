import numpy as np
import csv
import sys
import os
from defination import fvecs_write



dir = "/path/wikipedia/image_data_test/resnet_embeddings"   # need modify
root = "/path/wikipedia10m/"  # need modify
save_data_file = root + "data.fvecs"
save_train_file = root + "train.fvecs"
save_query_file = root + "query.fvecs"
file_names = os.listdir(dir)
os.makedirs(root, exist_ok=True)

data_size = 10000000 # 10m
train_size = 100000
query_size = 10000

dim = 2048
embeddings = []

index = 0
for file_name in file_names:
    file_path = os.path.join(dir, file_name)
    
    # Check if the file is a CSV file
    if file_name.endswith('.csv'):
        print(f"Reading file: {file_name}")
        
        # Open and read the CSV file
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file, delimiter='\t')
            for row in csv_reader:
                embedding = row[1].split(',')
                assert len(embedding) == dim
                embedding = [float(val) for val in embedding]
                embeddings.append(embedding)
                if len(embeddings) == data_size+query_size:
                    break
    if len(embeddings) == data_size+query_size:
        break


fvecs = np.array(embeddings, dtype=np.float32)
print("shape:", fvecs.shape)
data = fvecs[:data_size]
query = fvecs[data_size:data_size+query_size]
train_indices = np.random.choice(data_size, size=train_size, replace=False)
train = data[train_indices]



# Save the integer vectors to a binary file
fvecs_write(save_data_file, data)
fvecs_write(save_train_file, train)
fvecs_write(save_query_file, query)
print("Done!")
