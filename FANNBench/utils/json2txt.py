#used to convert json attributions into typical format for DiskANN

import json
import numpy as np
import csv
import sys
from defination import check_file, check_dir


if __name__ == "__main__":
    if not len(sys.argv) == 3 :
        print("error wrong argument")
        exit()
    else:
        attr_file = sys.argv[1]
        print("input file:", attr_file)
        check_file(attr_file)

        output_attr_new_file = sys.argv[2]
        print("output file:", output_attr_new_file)
        check_dir(output_attr_new_file)

    #json: [1,2,3]
    #txt:  1\n2\n3

    #read
    with open(attr_file, 'r') as file:
        data = json.load(file)
    assert(isinstance(data, list))
    data = np.array(data)

    #write
    with open(output_attr_new_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in data:
            writer.writerow([item])

    print("succeed to transport ", attr_file, " into ", output_attr_new_file, " for DiskANN label file")
