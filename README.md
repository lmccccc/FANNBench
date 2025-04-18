# FANN Bench
Unified interface for Filtering Approximate Nearest Neighbor (Filtering ANN) search.

## 📚 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1.ACORN](#1-ACORN-installation)
  - [2.DiskANN](#2-DiskANN-installation)
  - [3.DynamicSegmentGraph(DSG)](#3-DSG-installation)
  - [4.Faiss](#4-faiss-installation)
  - [5.iRangeGraph](#5-iRangeGraph-installation)
  - [6.RangeFilteredANN(beta-WST)](#6-WST-installation)
  - [7.SeRF](#7-SeRF-installation)
  - [8.UNIFY](#8-UNIFY-installation)
  - [9.Milvus](#9-Milvus-installation)
  - [10.NHQ](#10-NHQ-NSW)

- [Usage](#usage)
  - [1.Data preparison](#1-data-preparison)
  - [2.Index construction](#index-construction)
  - [3.Query](#query)
  - [3.Data processing](#data-processing)
- [Source Code Reference (Optional)](#-source-code-reference-optional)
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

| Dataset | Link |
|---------|------|
| SIFT | http://corpus-texmex.irisa.fr/ |
| Spacev | https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B |
| Redcaps | https://redcaps.xyz/ |
| Youtube | https://research.google.com/youtube8m/download.html |

We assume that you downloaded all dataset we need.

#### 1.1 SIFT

Modify FANNBench/utils/bvecs2fvecs.py line 19, 20, to align with your storage.
Run command
```bash
python  FANNBench/utils/bvecs2fvecs.py
```
#### 1.2 Spacev

Modify FANNBench/utils/i8bin2fvecs.py line 45, 46, to align with your storage.
Run command
```bash
python  FANNBench/utils/i8bin2fvecs.py
```

#### 1.3 Redcaps
Download redcaps can refer to RangeFilteredANN/generate_datasets/download_redcaps.py
Modify FANNBench/utils/npy2fvecs.py line 45, 46, to align with your storage.
Then run command.
```bash
python  FANNBench/utils/npy2fvecs.py
```

#### 1.4 Youtube
(1). Download Youtube json file
(2). Get google youtube key.
(3). Modify FANNBench/utils/get_youtube_attr.py line 14, 15.
(4). Modify FANNBench/utils/merge_youtube.py line 14, 15, 16.
(5). Run following command.
```bash
python  FANNBench/utils/get_youtube_attr.py
python  FANNBench/utils/merge_youtube.py
```
It takes days.


#### 1.2 Path configuration
Open FANNBench/vars.sh
Modify your dataset root in line 60, 70, 80 and 91, to align with your output path in step 1.1.

### 2. Attribute generation

#### 2.1 Numerical attribute for range query
```bash
cd FANNBench
# configure var.sh
python utils/modify_var.py label_range 100000
python utils/modify_var.py label_cnt 1
python utils/modify_var.py query_label_cnt 6
python utils/modify_var.py query_label 0
# generate attribue, query range, and ground truth
./run_attr_generator.sh
./run_qrange_generator.sh
./run_groundtruth_generator.sh
```

#### 2.2 Categorical attribute for label query
```bash
cd FANNBench
# configure var.sh
python utils/modify_var.py label_range 500
python utils/modify_var.py label_cnt 1
python utils/modify_var.py query_label_cnt 1
python utils/modify_var.py query_label 6
# generate attribue, query range, and ground truth
./run_attr_generator.sh
./run_qrange_generator.sh
./run_groundtruth_generator.sh
```

### 2. Index construction

Before constructing Milvus index, make sure Milvus docker service is up.
Generate index one by one. Before building index, make sure configuration match its filtering strategy.
In var.sh, for range filtering:
```bash
label_range=100000
query_label_cnt=6 (6, 10, 19, or 20, representing for 50%, 10%, 1% and 0.1% selectivity)
query_label=0
```
For label filtering:
```bash
label_range=500
query_label_cnt=1
query_label=6 (6, 10, 19, or 20, representing for 50%, 10%, 1% and 0.1% selectivity)
```
Construction:
```bash
cd FANNBench
./run_hnsw.sh construction                  # Faiss-HNSW
./run_ivfpq.sh construction                 # Faiss-IVFPQ
./run_milvus_hnsw.sh construction           # Milvus-HNSW
./run_milvus_ivfpq.sh construction          # Milvus-IVFPQ
./run_acorn.sh construction                 # ACORN
./run_serf.sh construction                  # SeRF
./run_dsg.sh construction                   # DSG
./run_irange.sh construction                # iRangeGraph
./run_wst.sh construction                   # WST-opt
./run_vamanatree.sh construction            # WST-Vamana
./run_unify.sh construction                 # UNIFY-CBO
./run_unify_hybrid.sh construction          # UNIFY-joint
./run_diskann.sh construction               # FDiskANN-VG
./run_diskann_stitched.sh construction      # FDiskANN-SVG
./run_nhq_nsw.sh construction               # NHQ-NSW
./run_nhq_kgraph.sh construction            # NHQ-KGraph
```

### 3. Query

#### 3.1 Single query
If query for single algorithm at single param (like ef_search=150)
Modify params in var.sh, then
```bash
./run_xxx.sh query
```
#### 3.2 Batch param query
```bash
./all_query.sh 'algo'
```
Avaliable 'algo': acorn(ACORN), diskann(FDiskANN-VG), diskann_stitched(FDiskANN-SVG), hnsw(Faiss-HNSW), irange(iRangeGraph), ivfpq(Faiss-IVFPQ), milvus_ivfpq(Milvus-IVFPQ), milvus_hnsw(Milvus-HNSW), kgraph(NHQ-KGraph), nsw(NHQ-NSW), serf(SeRF), dsg(DSG), vamana_tree(WST-Vamana), wst_sup_opt(WST-opt), unify(UNIFY-CBO), unify_hybrid(UNIFY-joint).

#### 3.3 Cross selectivity experiments
```bash
./all_query.sh batch 'algo'
```
It support search for 0.1%, 1%, 10% and 50% at once.


### 4. Data processing

```bash
./run_plot.sh qpsbar  # plot for range query QPS bar at 90% recall
./run_plot.sh qpsbarlabel  # plot for label query QPS bar at 90% recall
./run_plot.sh index # print index size and construction time, and query memory usage

```

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


## Licnese

This project is licensed under the [MIT License](./LICENSE).