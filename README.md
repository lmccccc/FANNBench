### FANN Bench
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
  - [2.Index construction](#index construction)
  - [3.Query](#query)
  - [3.Data processing](#data processing)
- [License](#license)

---
## ✅ Prerequisites

Make sure you have the following installed:
- Python v3.8
- Docker
- BLAS

---

## 🔧 Installation

```bash
git clone https://anonymous.4open.science/r/FANNBench-41C0
cd FANNBench
```

### 1. ACORN installation

```bash
cd ACORN
cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_C_API=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2
make -C build -j faiss
make -C build acorn_build
make -C build acorn_query
```
### 2. DiskANN installation

```bash
cd DiskANN
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 
```

### 3. DSG installation

```bash
cd DynamicSegmentGraph
mkdir build && cd build
cmake ..
make
```

### 4. Faiss installation

```bash
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

```bash
mkdir build && cd build && cmake .. && make
```

### 6. beta-WST installation

```bash
cd RangeFilteredANN
pip3 install .
```

### 7. SeRF installation

```bash
cd SeRF
mkdir build && cd build
cmake ..
make
```

### 8. UNIFY installation

```bash
cd python_bindings
python setup.py install
```

### 8. Milvus installation

Milvus Standalone runs using Docker, and users need to download it manually.
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.5.9/milvus-standalone-docker-compose.yml -O docker-compose.yml
sudo docker compose up -d
```
---
## 🚀 Usage

### 1. Data preparision

### 2. Index construction

### 3. Query

### 4. Data processing

---
## 📁 Source Code Reference (Optional)

You do **not** need to use or modify the source code directly.
But if you are curious, here is where each app lives in the repo:

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