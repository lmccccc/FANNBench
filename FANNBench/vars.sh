# vars.sh

# lable generation
 #keyword or range for query.
label_method=keyword # don't change
query_method=keyword # don't change
distribution=random  # random, in_dist, out_dist
label_range=2        # 2, 12, 128
query_label_cnt=1    # 1

# sel_{query label cnt}_{total label cnt}_{generation method}
label_attr=sel_${query_label_cnt}_${label_range}_${distribution} # to name different label files

# common vars
ef_construction=1000
M=40 # normorlly 40, but 8 in serf
serf_M=8
K=10
gt_topk=${K}
nprobe=10
ef_search=100

# threads
threads=128

#acorn vars
gamma=12
M_beta=64

# disk ann typical variables
# FilteredLbuild=12
alpha=1.2
L="100 400" # "10 20 30 40 50 100"

#rii an pq based
partition_size_M=128

#RangeFilteredANN, super opt postfiltering
beamsize=10
final_beam_multiply=16

# dataset vars

#siftsmall（10k）
# N=10000
# query_size=100
# dataset=siftsmall

# root="/mnt/data/dataset/siftsmall/" 
# dataset_file=${root}siftsmall_base.fvecs
# query_file=${root}siftsmall_query.fvecs
# dataset_attr_file=${root}siftsmall_attr.json
# query_range_file=${root}siftsmall_qrange.json
# ground_truth_file=${root}siftsmall_gt_${gt_topk}.json
# train_file=${root}siftsmall_learn.fvecs

# sift1m
# N=1000000
# query_size=10000
# train_size=100000
# dataset=sift
# root="/mnt/data/mocheng/dataset/sift/" 

# dataset_file=${root}sift_base.fvecs
# query_file=${root}sift_query.fvecs
# train_file=${root}sift_learn.fvecs

# dataset_bin_file=${root}data_base.bin # to be generated
# query_bin_file=${root}data_query.bin  # to be generated

# sift1b
# dataset_file=${root}bigann_base.bvecs
# query_file=${root}bigann_query.bvecs
# dataset_attr_file=${root}sift_attr.json
# query_range_file=${root}sift_qrange.json
# ground_truth_file=${root}sift_gt_10.json


# deep1b
N=1000000000
query_size=10000
train_size=100000
dataset=deep

root="/mnt/data/mocheng/dataset/deep/"
dataset_bin_file=${root}base.1B.fbin
query_bin_file=${root}query.public.10K.fbin
  
dataset_file=${root}base.1B.fvecs        # to be generated
query_file=${root}query.public.10K.fvecs # to be generated
train_file=${root}deep_learn100K.fvecs   # to be generated
python utils/fbin2fvecs.py ${dataset_bin_file} ${dataset_file} ${N}
python utils/fbin2fvecs.py ${query_bin_file} ${query_file} ${query_size}
python utils/generate_train.py ${dataset_file} ${train_file} ${train_size}



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

#RangeFilteredANN file path
rfann_root=${root}rfann/
rfann_index_root=${rfann_root}index/
rfann_index_prefix=${rfann_index_root}index_${label_attr}/
rfann_result_root=${rfann_root}result/
rfann_result_file=${rfann_result_root}result__${label_attr}.csv


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

#nhq
nhq_root=${root}nhq/
nhq_index_root=${nhq_root}index/
nhq_index_model_file=${nhq_index_root}nhq_index_${label_attr}_M${M}_efc${ef_construction}
nhq_index_attr_file=${nhq_index_root}nhq_attr_${label_attr}_M${M}_efc${ef_construction}


