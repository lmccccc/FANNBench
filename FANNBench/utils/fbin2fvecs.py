#used to convert json attributions into typical format for DiskANN

import numpy as np
import sys
from defination import check_file, fvecs_write 
import struct
import os


def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)



if __name__ == "__main__":
    if not len(sys.argv) == 4 :
        print("error wrong argument size:", len(sys.argv))
        sys.exit(-1)
    else:
        fbin_file = sys.argv[1]
        check_file(fbin_file)

        fvecs_file = sys.argv[2]

        datasize = int(sys.argv[3])
        

    #json: [1,2,3]
    #txt:  1\n2\n3
    
    if os.path.isfile(fvecs_file):
        print(fvecs_file, " already exist")
        sys.exit(0)
        
    print("input file:", fbin_file)
    print("output file:", fvecs_file)
    print("datasize:", datasize)
    
    #read
    data = read_fbin(fbin_file)
    assert(data.shape[0] == datasize)

    #write
    fvecs_write(fvecs_file, data)
    

    print("succeed to transport ", fbin_file, " into ", fvecs_file)
