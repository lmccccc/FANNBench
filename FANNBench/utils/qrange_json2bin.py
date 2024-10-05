#used to convert json attributions into typical format for DiskANN

import numpy as np
import sys
from defination import check_file, check_dir, ivecs_read
import struct
import json


if __name__ == "__main__":
    if not len(sys.argv) == 3 :
        print("error wrong argument size:", len(sys.argv))
        sys.exit(-1)
    else:
        qrange_file = sys.argv[1]
        print("input file:", qrange_file)
        check_file(qrange_file)

        output_qrange_file = sys.argv[2]
        print("output file:", output_qrange_file)
        check_dir(output_qrange_file)
        

    #json: [1,2,3]
    #txt:  1\n2\n3

    #read
    with open(qrange_file, 'r') as file:
        data = json.load(file)
    assert(isinstance(data, list))
    assert(len(data) > 0)
    assert(isinstance(data[0], int))

    #write
    with open(output_qrange_file, mode='wb') as file:
        format_str = f'{len(data)}i'
        list_pack = struct.pack(format_str, *data)
        file.write(list_pack)

    print("succeed to transport ", qrange_file, " into ", output_qrange_file, " for iRangeGraph query range file")
