# UNIFY - Unified Index for Range Filtered Approximate Nearest Neighbors Search
This repo is the implementation of UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search.

Header-only C++ implementation for HSIG, with python bindings.

## Quick Start

### Compile

```bash
cd python_bindings
python setup.py install
``` 

### Datasets
The datasets can be downloaded from their official websites, and users can generate attributes and query ranges as the following: 
```bash
cd benchmark
cd code
python preprocessing.py
``` 

### Running example benchmark:
```bash
cd benchmark
cd code
python search_hsig.py --use_mbv_hnsw true --data_path YOUR_DATA_DIR --index_cache_path YOUR_INDEX_DIR --result_save_path YOUR_RESULT_PATH
```




