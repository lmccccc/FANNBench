# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import json

# https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
logging.basicConfig()
logger = logging.getLogger('faiss.contrib.exhaustive_search')
logger.setLevel(logging.INFO)

from faiss.contrib import datasets
from faiss.contrib.exhaustive_search import knn_ground_truth
from faiss.contrib import vecs_io

ds = datasets.DatasetDeep1B(nb=int(1e9))

json_file = 'testing_data/sift_attr.json'
with open(json_file) as file:
    data = json.load(file)
print(data[:3])

test_k = 10


print("computing GT matches for", ds)

D, I = knn_ground_truth(
    ds.get_queries(),
    ds.database_iterator(bs=65536),
    k=500
)


output_file = 'testing_data/sift_gt_5.json'
# vecs_io.ivecs_write("/tmp/tt.ivecs", I)
