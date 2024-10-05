#used to convert json attributions into typical format for DiskANN

import numpy as np
import sys
from defination import check_file, fvecs_write , fvecs_read
import struct
import os




if __name__ == "__main__":
    if not len(sys.argv) == 4 :
        print("error wrong argument size:", len(sys.argv))
        sys.exit(-1)
    else:
        fvecs_file = sys.argv[1]
        check_file(fvecs_file)

        train_file = sys.argv[2]

        train_size = int(sys.argv[3])
        
    
    if os.path.isfile(train_file):
        print(train_file, " already exist")
        sys.exit(0)
        
    print("input file:", fvecs_file)
    print("output file:", train_file)
    print("train size:", train_size)
        
    #read
    data = fvecs_read(fvecs_file)

    np.random.seed(42)
    data_rows = data.shape[0]
    choose_idxs = np.random.choice(data_rows, size=train_size, replace=False)
    train = data[choose_idxs, :]

    #write
    fvecs_write(train_file, train)

    print("succeed to generate train file ", train_file)
