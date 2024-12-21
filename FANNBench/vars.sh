# vars.sh


# label range   label_cnt   query_label_cnt   query_label
# <500          1           1                 useless                    # For all methods, query random {1} label for each vector has {1} label too.
# <500          >1          1                 0 ~ label_cnt-1            # For keyword query, query {query_label} for vectors which have at most {label_cnt} labels (vector label is in zipf distribution).
# >0            1           >1                useless                    # For range query, query range [random, random+{query_label_cnt}], each vector has {1} label.
# <500          >1          $label_cnt        useless                    # For nhq_kgraph and nhq_nsw, same dimension for label and query label

# lable generation
distribution=in_dist  # random, in_dist, out_dist
label_range=50000         # keyword query shall less than 500

# keyword or range for query. Only one of them can >1 at once
label_cnt=1           # 1 or $label_range if $labelrange (keyword), not suitable for: 1.acorn,    2.irange,  3.vamanatree,  4.wst 5.nhq_nsw, 6.nhq_kgraph
query_label_cnt=1000     # 1 or more         if >1 (range query for [x, x+query_label_cnt]), not suitable for: 1.diskann,  2.nhq_nsw, 3.nhq_kgraph
query_label=1         # work only when label_cnt > 1



# milvus, hnsw, ivfpq, rii are subset search, which means they are suitable for both range and keyword query
# for batch query at different efs/nprobe/beamsize, see run_multi_query.sh

# var list         construction                                                      search
# acorn          M M_beta                                                          ef_search
# diskann        M alpha                                                           L
# hnsw           M ef_construction                                                 ef_search
# irange         M ef_construction                                                 (M) ef_search
# ivfpq          partition_size_M                                                  nprobe
# milvus_ivfpq   partition_size_M                                                  nprobe
# nhq_kgraph     kgraph_L iter S R RANGE(alias M) PL(alias ef_cons) B kgraph_M     weight_search L_search(alias ef_search)
# nhq_nsw        M ef_construction                                                 ef_search
# rii            partition_size_M
# serf           serf_M ef_construction                                            ef_search
# vamana_tree    split_factor                                                      beamsize
# wst            split_factor shift_factor                                         beamsize final_beam_multiply

# other vars are fixed inside the code

# hnsw vars
M=40 # normarlly 40
ef_construction=1000
ef_search=400

#serf vars
serf_M=8 # fixed

#acorn vars
gamma=12
M_beta=64

# DiskAnn typical                                                                                                     
# FilteredLbuild=12
alpha=1.2 # fixed
L=400 # "10 20 30 40 50 100" efsearch

#rii an pq based
partition_size_M=64
nprobe=10

#RangeFilteredANN, super opt postfiltering
beamsize=40
split_factor=2
shift_factor=0.5
final_beam_multiply=16

#nhq kgraph vars
kgraph_L=100  # <L> is the parameter controlling the graph quality, larger is more accurate but slower, no smaller than K.
iter=12       # <iter> is the parameter controlling the maximum iteration times, iter usually < 30.
S=10          # <S> is the parameter contollling the graph quality, larger is more accurate but slower.
R=300         # <R> is the parameter controlling the graph quality, larger is more accurate but slower.
RANGE=${M}      # ${M}  # <RANGE> controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.
PL=${ef_construction}        # ${ef_construction}     # <PL> controls the quality of the NHQ-NPG_kgraph, the larger the better.
B=0.4         # <B> controls the quality of the NHQ-NPG_kgraph.
kgraph_M=1    # <M> controls the edge selection of NHQ-NPG_kgraph.

weight_search=140000
L_search=${ef_search}

# common vars
# label_method=keyword # don't change
# query_method=keyword # don't change
K=10
gt_topk=${K}

# threads
threads=128

# Check if the user provided the required input
if [ $# -lt 1 ]; then
    echo "Usage: $0 <construction|query> option: multi_ vars"
    exit 1
fi

# Get the input argument
mode=$1
multi=$2
# Set thread number based on the mode
if [ "$mode" == "construction" ]; then
    threads=128
elif [ "$mode" == "query" ]; then
    threads=1
    if [ "$multi" == "multi_hnsw" ]; then
        multi=multi
        ef_search=$3
    elif [ "$multi" == "multi_diskann" ]; then
        multi=multi
        L=$3
    elif [ "$multi" == "multi_ivfpq" ]; then
        multi=multi
        nprobe=$3
    elif [ "$multi" == "multi_kgraph" ]; then
        multi=multi
        ef_search=$3
        L_search=$3
    elif [ "$multi" == "multi_vtree" ]; then
        multi=multi
        beamsize=$3
    elif [ "$multi" == "multi_wst" ]; then
        multi=multi
        beamsize=$3
        final_beam_multiply=$4
    fi
elif [ "$mode" == "all" ]; then
    threads=128
else
    echo "Invalid mode. Please use 'construction' or 'query'."
    exit 1
fi

label_attr=sel_${query_label_cnt}_${label_cnt}_${label_range}_${distribution} # to name different label files
result_file=exp_results.csv

# dataset vars

#siftsmall（10k）
# N=10000
# query_size=100
# dataset=siftsmall

# root="/mnt/data/mocheng/dataset/siftsmall/" 
# dataset_file=${root}siftsmall_base.fvecs
# query_file=${root}siftsmall_query.fvecs
# dataset_attr_file=${root}siftsmall_attr.json
# query_range_file=${root}siftsmall_qrange.json
# ground_truth_file=${root}siftsmall_gt_${gt_topk}.json
# train_file=${root}siftsmall_learn.fvecs

# sift1m
dim=128
N=1000000
query_size=10000
train_size=100000
dataset=sift1M
root="/mnt/data/mocheng/dataset/sift/" 

dataset_file=${root}sift_base.fvecs
query_file=${root}sift_query.fvecs
train_file=${root}sift_learn.fvecs

dataset_bin_file=${root}data_base.bin # to be generated
query_bin_file=${root}data_query.bin  # to be generated

# sift1b
# dataset_file=${root}bigann_base.bvecs
# query_file=${root}bigann_query.bvecs
# dataset_attr_file=${root}sift_attr.json
# query_range_file=${root}sift_qrange.json
# ground_truth_file=${root}sift_gt_10.json


# deep1b
# dim=96
# N=1000000000
# query_size=10000
# train_size=100000
# dataset=deep

# root="/mnt/data/mocheng/dataset/deep/"
# dataset_bin_file=${root}base.1B.fbin
# query_bin_file=${root}query.public.10K.fbin
  
# dataset_file=${root}base.1B.fvecs        # to be generated
# query_file=${root}query.public.10K.fvecs # to be generated
# train_file=${root}deep_learn100K.fvecs   # to be generated
# python utils/fbin2fvecs.py ${dataset_bin_file} ${dataset_file} ${N}
# python utils/fbin2fvecs.py ${query_bin_file} ${query_file} ${query_size}
# python utils/generate_train.py ${dataset_file} ${train_file} ${train_size}

# deep10m
# dim=96
# N=10000000
# query_size=10000
# train_size=10000
# dataset=deep10m

# root="/mnt/data/mocheng/dataset/deep10m/"
# dataset_bin_file=${root}base.10M.fbin
# query_bin_file=${root}query.public.10K.fbin
  
# dataset_file=${root}base.10M.fvecs        # to be generated
# query_file=${root}query.public.10K.fvecs # to be generated
# train_file=${root}deep_learn100K.fvecs   # to be generated
# python utils/fbin2fvecs.py ${dataset_bin_file} ${dataset_file} ${N}
# python utils/fbin2fvecs.py ${query_bin_file} ${query_file} ${query_size}
# python utils/generate_train.py ${dataset_file} ${train_file} ${train_size}



# label file, the name of path is defined in label_attr
label_path=${root}${label_attr}/
dataset_attr_file=${label_path}attr_${label_attr}.json
query_range_file=${label_path}qrange${label_attr}.json
ground_truth_file=${label_path}sif_gt_${gt_topk}.json
centroid_file=${label_path}centroid.npy

# milvus_ivfpq collection name
collection_name=collection_${dataset}_${label_attr}
milvus_coll_path=/var/lib/milvus/etcd/data/${collection_name}/   # useless

# milvus_hnsw collection name
hnsw_collection_name=collection_hnsw_${dataset}_${label_attr}_efc${ef_construction}
milvus_hnsw_coll_path=/var/lib/milvus/etcd/data/${hnsw_collection_name}/    # useless



#acorn res path
acorn_root=${root}acorn/
acorn_index_root=${acorn_root}index/
acorn_index_file=${acorn_index_root}index_acorn_${label_attr}_M${M}_ga${gamma}_Mb${M_beta}
# acorn_result_root=${acorn_root}result/

#ivfpq res path
ivfpq_root=${root}ivfpq/
ivfpq_index_root=${ivfpq_root}index/
ivfpq_index_file=${ivfpq_index_root}index_ivfpq_${label_attr}_${partition_size_M}

#hnsw res path
hnsw_root=${root}hnsw/
hnsw_index_root=${hnsw_root}index/
hnsw_index_file=${hnsw_index_root}index_hnsw_${label_attr}_${M}_${ef_construction}

#DiskAnn need to transform fvecs to bin, and its index/result path

diskann_root=${root}diskann/
diskann_index_root=${diskann_root}index/
diskann_index_label_root=${diskann_index_root}index_diskann_${label_attr}_M${M}_efc${ef_construction}/
diskann_index_file=${diskann_index_label_root}index_diskann

diskann_result_root=${diskann_root}result/
diskann_result_path=${diskann_result_root}result_${label_attr}_efs${ef_search}

keyword_query_range_file=${label_path}sift_qrange_keyword.txt #only key word query supported
label_file=${label_path}data_attr.txt
ground_truth_bin_file=${label_path}sift_gt_${gt_topk}.bin

#RangeFilteredANN(WST super optimized postfiltering) file path
rfann_root=${root}rfann/
rfann_index_root=${rfann_root}index/
rfann_index_prefix=${rfann_index_root}index_${label_attr}/
rfann_result_root=${rfann_root}result/
rfann_result_file=${rfann_result_root}result_${label_attr}.csv

#RangeFilteredANN(vamana tree) file path
vtree_root=${root}vtree/
vtree_index_root=${vtree_root}index/
vtree_index_prefix=${vtree_index_root}index_${label_attr}/
vtree_result_root=${vtree_root}result/
vtree_result_file=${vtree_result_root}result_${label_attr}.csv

#iRangeGraph file path
irange_root=${root}irange/
irange_index_root=${irange_root}index/
irange_id2od_file=${irange_index_root}data_id2od_${label_attr}
irange_index_file=${irange_index_root}data_irange_${label_attr}
irange_result_root=${irange_root}result/
irange_result_file=${irange_result_root}result_${label_attr}

qrange_bin_file=${label_path}data_qrange.bin #format: left, rithgt, left, right...
attr_bin_file=${label_path}data_attr.bin

#rii
rii_root=${root}rii/
rii_index_root=${rii_root}index/
rii_index_file=${rii_index_root}index_rii_${partition_size_M}_${label_attr}

#serf
serf_root=${root}serf/
serf_index_root=${serf_root}serf_index/
serf_index_file=${serf_index_root}index_serf_M${M}_efs${ef_construction}_${label_attr}

#nhq_nsw
nhq_root=${root}nhq/
nhq_index_root=${nhq_root}index/
nhq_index_model_file=${nhq_index_root}nhq_index_${label_attr}_M${M}_efc${ef_construction}
nhq_index_attr_file=${nhq_index_root}nhq_attr_${label_attr}_M${M}_efc${ef_construction}


#nhq_kgraph
nhqkg_root=${root}nhqkg/
nhqkg_index_root=${nhqkg_root}index/
nhqkg_index_model_file=${nhqkg_index_root}nhqkg_index_${label_attr}_L${kgraph_L}_iter${iter}S${S}_R${R}_RANGE${RANGE}_PL${PL}_B${B}_M${kgraph_M}
nhqkg_index_attr_file=${nhqkg_index_root}nhqkg_attr_${label_attr}_L${kgraph_L}_iter${iter}S${S}_R${R}_RANGE${RANGE}_PL${PL}_B${B}_M${kgraph_M}


