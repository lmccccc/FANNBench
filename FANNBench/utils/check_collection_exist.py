
import sys
from pymilvus import DataType, MilvusClient
import sys
from defination import check_dir, check_file, ivecs_read, fvecs_read, bvecs_read, read_attr, read_file, write_attr_json
import numpy as np
import math
import time

# read args
if __name__ == "__main__":
    c_name = sys.argv[1]


    client = MilvusClient(
        uri="http://localhost:19530"
    )

    if not client.has_collection(c_name):
        print("collection ", c_name, " not exists")

    exit(0)
