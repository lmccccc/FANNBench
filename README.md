### Hello Runoob!# FANN Bench
Unified interface for Filtering Approximate Nearest Neighbor (Filtering ANN) search.

## 📚 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1.ACORN](#ACORN-installation)
  - [2.DiskANN](#DiskANN-installation)
  - [3.DynamicSegmentGraph(DSG)](#DSG-installation)
  - [4.Faiss](#faiss-installation)
  - [5.iRangeGraph](#iRangeGraph-installation)
  - [6.RangeFilteredANN(beta-WST)](#WST-installation)
  - [7.SeRF](#SeRF-installation)
  - [8.UNIFY](#UNIFY-installation)
  - [9.Milvus](#Milvus-installation)
  - [10.NHQ](#NHQ-NSW)
  
- [Usage](#usage)
  - [1.Data preparison](#Data preparison)   
  - [2.Run](#run)
  - [3.Data processing](#process)
- [License](#license)

---
## ✅ Prerequisites

Make sure you have the following installed:
- Python v3.8
- Docker
- BLAS
- gcc 13.3
- shell

---

## 🔧 Installation

git clone https://anonymous.4open.science/r/FANNBench-41C0
cd FANNBench

### 1. ACORN installation

```
cd ACORN
cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_C_API=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2
make -C build -j faiss
make -C build acorn_build
make -C build acorn_query
```
### 2. DiskANN installation

```
cd DiskANN
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 
```

### 3. DSG installation

```
cd DynamicSegmentGraph
mkdir build && cd build
cmake ..
make
```

### 4. Faiss installation

```
cd faiss
cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_C_API=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2
make -C build -j faiss
make -C build generate_groundtruth
make -C build hnsw_build
make -C build hnsw_query
make -C build ivfpq_build
make -C build ivfpq_query
```

### 5. iRangeGraph installation

```
mkdir build && cd build && cmake .. && make
```

## 📁 Source Code Reference (Optional)

You do **not** need to use or modify the source code directly.  
But if you're curious, here's where each app lives in the repo:

| App | Directory Link |
|-----|----------------|
| ACORN | https://github.com/stanford-futuredata/ACORN |
| DiskANN | https://github.com/microsoft/DiskANN |
| DynamicSegmentGraph(DSG) | https://github.com/rutgers-db/DynamicSegmentGraph/ |
| Faiss | https://github.com/facebookresearch/faiss |
| iRangeGraph | https://github.com/YuexuanXu7/iRangeGraph |
| SeRF | https://github.com/rutgers-db/SeRF |
| UNIFY | https://github.com/sjtu-dbgroup/UNIFY |
| Milvus | https://github.com/milvus-io/milvus |
| NHQ | Not avaliable |