# vars.sh

# lable generation
 #keyword or range for query.
label_method=keyword # don't change
query_method=keyword # don't change
distribution=random  # random, in_dist, out_dist
label_range=8        # 2, 12, 128
query_label_cnt=1    # 1

# sel_{query label cnt}_{total label cnt}_{generation method}
label_attr=sel_${query_label_cnt}_${label_range}_${distribution} # to name different label files
result_file=exp_results.csv

# common vars
K=10
gt_topk=${K}

# hnsw vars
M=40 # normarlly 40
ef_construction=1000
ef_search=400

#serf vars
serf_M=8 # stay fixed

# threads
threads=128

#acorn vars
gamma=12
M_beta=64

# DiskAnn typical                                                                                                     
# FilteredLbuild=12
alpha=1.2 # fixed
L=400 # "10 20 30 40 50 100" efsearch

#rii an pq based
partition_size_M=4
nprobe=100

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