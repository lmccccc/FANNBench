#used to convert json attributions into typical format for DiskANN
#ori: [2, 3, 1, 4], which means [2, 3] for query1, [1, 4] for query2
#output: 2,3\n 1,2,3,4\n
import json
import numpy as np
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


    #read
    with open(attr_file, 'r') as file:
        data = json.load(file)
    assert(isinstance(data, list))
    data = np.array(data).reshape((-1, 2))

    #write
    print_idx = 0
    with open(output_attr_new_file, mode='w') as file:
        for item in data:
            start = item[0]
            end = item[1]
            val = [i for i in range(start, end+1)]
            # if(print_idx < 5):
            #     print(print_idx , " label: ", val)
            #     print_idx += 1
            for val in range(start, end+1):
                file.write(str(val))
                if val != end:
                    file.write(', ')
            file.write('\n')

    print("succeed to transport ", attr_file, " into ", output_attr_new_file, " for DiskANN label file")
