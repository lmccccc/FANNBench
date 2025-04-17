# vars.sh


# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------- attribute var part  -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# label_range: the range of label, normally 100000 for range query, 500 for keyword query
# label_cnt: label cardinality, normally 1 for any query
# query_label_cnt: query range, normally label_range*selectivity for range query, 1 for keyword query
# query_label: query label, 0 for range query means inactivate, 1~19 for keyword query, means different selectivity's label

# label range   label_cnt     query_label_cnt   query_label
# 100000        1             1~20              0(inactivate)              # For range query, query range [random, random+{query_label_cnt}], each vector has {1} label.
# 500           1             1                 1~20                       # for label query for both diskann and nhq_nsw.(range filtering can do it too. But not necessary to compare)
# lable generation
distribution=random     # random, in_dist, out_dist, raeal
label_range=500

# keyword or range for query. Only one of them can >1 at once
label_cnt=1
# qrange
query_label_cnt=1
query_label=6

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------- dataset var part  ---------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---begin dataset---

## sift10m
# dataset=sift10M

## spacev10m
# dataset=spacev10m

## redcaps1m
# dataset=redcaps1m

## YT-RGB1m
dataset=YTRGB1m

## sift1M
# dataset=sift1M

# ---end dataset---





if [  "$dataset" == "sift10M" ]; then
    dim=128
    N=10000000
    query_size=10000
    train_size=1000000
    root="/your_path/dataset/sift10m/"         # need modify
    dataset_file=${root}sift10m.fvecs
    query_file=${root}sift10m_query.fvecs
    train_file=${root}sift10m_train.fvecs
    distribution=random
elif [  "$dataset" == "spacev10m" ]; then
    dim=100
    N=10000000
    query_size=10000
    train_size=1000000
    root="/your_path/dataset/spacev10m/"        # need modify
    dataset_file=${root}base10M.fvecs
    query_file=${root}query10k.fvecs
    train_file=${root}train.fvecs
    distribution=random
elif [  "$dataset" == "redcaps1m" ]; then
    dim=512
    N=1000000
    query_size=10000
    train_size=100000
    root="/your_path/dataset/redcaps1m/"        # need modify
    dataset_file=${root}image_embeddings.fvecs
    query_file=${root}query.fvecs
    train_file=${root}train.fvecs
    real_attr_file=${root}timestamp.json
    distribution=real
elif [  "$dataset" == "YTRGB1m" ]; then
    dim=1024
    N=1000000
    query_size=10000
    train_size=1000000
    root="/your_path/dataset/youtube1m/"         # need modify
    dataset_file=${root}rgb.fvecs
    query_file=${root}rgb_query.fvecs
    train_file=${root}rgb_train.fvecs
    real_attr_file=${root}views.json
    distribution=real
elif [ "$dataset" == "sift1M" ]; then
    dim=128
    N=1000000
    query_size=10000
    train_size=100000
    root="/your_path/dataset/sift/"         # need modify
    dataset_file=${root}sift_base.fvecs
    query_file=${root}sift_query.fvecs
    train_file=${root}sift_learn.fvecs
fi

if [ $query_label -gt 0 ]; then
    distribution=random
fi
    


## siftsmall（10k）
# dataset=siftsmall
# dim=128
# N=10000
# query_size=100
# train_size=10000
# root="/your_path/dataset/siftsmall/" 
# dataset_file=${root}siftsmall_base.fvecs
# query_file=${root}siftsmall_query.fvecs
# train_file=${root}siftsmall_learn.fvecs

## sift1m
# dataset=sift1M
# dim=128
# N=1000000
# query_size=10000
# train_size=100000
# root="/your_path/dataset/sift/" 
# dataset_file=${root}sift_base.fvecs
# query_file=${root}sift_query.fvecs
# train_file=${root}sift_learn.fvecs

## yfcc10m
# dataset=yfcc10m
# dim=192
# N=10000000
# query_size=10000
# train_size=100000
# root="/your_path/dataset/yfcc10m/"
# dataset_file=${root}base10M.fvecs
# query_file=${root}query10k.fvecs
# train_file=${root}train.fvecs
# distribution=random

## deep10m
# dataset=deep10m
# dim=96
# N=10000000
# query_size=10000
# train_size=10000
# root="/your_path/dataset/deep10m/"
# dataset_file=${root}base.10M.fvecs      
# query_file=${root}query.public.10K.fvecs
# train_file=${root}deep_learn100K.fvecs  
# distribution=random


# # for all dataset
dataset_bin_file=${root}data_base.bin # to be generated
query_bin_file=${root}data_query.bin  # to be generated


# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------- hyper parameter var part  -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# milvus, hnsw, ivfpq, rii are subset search, which means they are suitable for both range and keyword query
# for batch query at different efs/nprobe/beamsize, see run_multi_query.sh

# var list              construction                                                         search
# acorn                   M M_beta gamma                                                    ef_search
# diskann_memory          M alpha                                                           L
# diskann_stitched        M alpha Stitched_R                                                L
# hnsw                    M ef_construction                                                 ef_search
# irange                  M ef_construction                                                 (M) ef_search
# ivfpq                   partition_size_M                                                  nprobe
# milvus_ivfpq            partition_size_M                                                  nprobe
# nhq_kgraph              kgraph_L iter S R RANGE(alias M) PL(alias ef_cons) B kgraph_M     weight_search L_search(alias ef_search)
# nhq_nsw                 M ef_construction                                                 ef_search
# rii                     partition_size_M
# serf                    serf_M ef_max ef_construction                                     ef_search
# DSG                     serf_M ef_max ef_construction                                     ef_search
# WST_vamana              split_factor                                                      beamsize
# WST_opt                 split_factor shift_factor                                         beamsize final_beam_multiply
# UNIFY                   M ef_construction B_unify                                         ef_search AL

# other vars are fixed inside the code

# hnsw vars
M=40 # normarlly 40
ef_construction=1000
ef_search=200

#serf vars
serf_M=8

#dsg vars
ef_max=1000

#acorn vars
gamma=25
M_beta=64

# DiskAnn typical                                                                                                     
# FilteredLbuild=12
alpha=1.2 # fixed
L=1000 # "10 20 30 40 50 100" efsearch

# DiskANN Stitched
Stitched_R=80

#rii an pq based
partition_size_M=$dim
# $(($dim/2)) $dim
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

#UNIFY vars
B_unify=8
AL=16 # "[8, 16, 32]"

# airship: not used
alter_ratio=0.5

# common vars
# label_method=keyword # don't change
# query_method=keyword # don't change
K=10
gt_topk=${K}

# threads
threads=128

# all modify ends here, don't change the following code

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------- multi query parameter var part (no need to change)  -----------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# Check if the user provided the required input
if [ $# -lt 1 ]; then
    echo "Usage: $0 <construction|query|drop> option: multi_ vars"
    exit 1
fi

# Get the input argument
mode=$1
multi=$2
# Set thread number based on the mode
if [ "$mode" == "construction" ]; then
    if [ "$multi" == "multi_cons" ]; then
        distribution=$3
    fi
elif [ "$mode" == "drop" ]; then
    echo "drop only for miluvs, others use rm directly"
elif [ "$mode" == "query" ]; then
    threads=1

    if [ "$multi" == "multi_hnsw" ]; then
        multi=multi
        ef_search=$3
        if [ ! -z "$4" ]; then
            query_label_cnt=$4
            echo "query_label_cnt: $query_label_cnt"
        fi
    elif [ "$multi" == "multi_diskann" ]; then
        multi=multi
        L=$3
        if [ ! -z "$4" ]; then
            query_label_cnt=$4
            echo "query_label_cnt: $query_label_cnt"
        fi
    elif [ "$multi" == "multi_ivfpq" ]; then
        multi=multi
        nprobe=$3
        if [ ! -z "$4" ]; then
            query_label_cnt=$4
            echo "query_label_cnt: $query_label_cnt"
        fi
    elif [ "$multi" == "multi_kgraph" ]; then
        multi=multi
        ef_search=$3
        L_search=$3
        if [ ! -z "$4" ]; then
            query_label_cnt=$4
            echo "query_label_cnt: $query_label_cnt"
        fi
    elif [ "$multi" == "multi_vtree" ]; then
        multi=multi
        beamsize=$3
        if [ ! -z "$4" ]; then
            query_label_cnt=$4
            echo "query_label_cnt: $query_label_cnt"
        fi
    elif [ "$multi" == "multi_wst" ]; then
        multi=multi
        beamsize=$3
        final_beam_multiply=$4
        if [ ! -z "$5" ]; then
            query_label_cnt=$5
            echo "query_label_cnt: $query_label_cnt"
        fi
    elif [ "$multi" == "multi_unify" ]; then
        multi=multi
        AL=$3
        ef_search=$4
        if [ ! -z "$5" ]; then
            query_label_cnt=$5
            echo "query_label_cnt: $query_label_cnt"
        fi
    fi

else
    echo "Invalid mode. Please use 'construction' or 'query'."
    exit 1
fi

if [ "$multi" == "batch_qr" ] || [ "$multi" == "batch_gt" ]; then
    query_label_cnt=$3
fi

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------- attribute file part (no need to change)  ----------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

if [ "$distribution" == "in_dist" ] || [ "$distribution" == "out_dist" ] ; then
    index_dist=dist
else
    index_dist=$distribution
fi
label_attr=sel_${label_cnt}_${label_range}_${index_dist} # to name different label files
query_attr=sel_${query_label_cnt}_${label_cnt}_${label_range}_${distribution} # nothing with query
result_file=exp_results.csv

# if [ $label_cnt -gt 1 ]; then
#     query_attr=sel_${query_label_cnt}_${label_cnt}_${label_range}_${distribution}_${query_label} # diskann

#     if [ $label_cnt == $query_label_cnt ]; then # nhq
#         label_attr=sel_${query_label_cnt}_${label_cnt}_${label_range}_${index_dist}_${query_label} # to name different label files
#         query_attr=sel_${query_label_cnt}_${label_cnt}_${label_range}_${distribution}
#     fi
# fi

if [ $query_label -gt 0 ]; then
    echo "categorical index"
    # label_attr=sel_${label_cnt}_${label_range}_${distribution}_${query_label} # to name different label files
    query_attr=sel_${query_label_cnt}_${label_cnt}_${label_range}_${distribution}_${query_label}
fi





# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------- index file part (no need to change)  --------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# label file, the name of path is defined in label_attr
label_root=${root}label/
if [ ! -d "$label_root" ]; then
    mkdir ${label_root}
fi
label_path=$label_root${label_attr}/
if [ ! -d "$label_path" ]; then
    mkdir ${label_path}
fi
dataset_attr_file=${label_path}attr_${label_attr}.json
query_range_file=${label_path}qrange${query_attr}.json
ground_truth_file=${label_path}sif_gt_${query_attr}_${gt_topk}.json
centroid_file=${label_path}centroid.npy
qrange_bin_file=${label_path}data_qrange${query_attr}.bin #format: left, rithgt, left, right...
attr_bin_file=${label_path}data_attr.bin
ground_truth_bin_file=${label_path}sift_gt_${gt_topk}_${query_attr}.bin

if [ $query_label -gt 0 ]; then
    if [ "$distribution" == "in_dist" ] || [ "$distribution" == "out_dist" ] ; then
        dist_query_file=${label_path}query_${query_attr}.fvecs
        dist_query_size=$(expr $query_size / 2)
        ori_query_file=$query_file
        query_file=$dist_query_file
        query_bin_file=${label_path}query_${query_attr}.bin

        ori_query_size=$query_size
        query_size=$dist_query_size
    fi
fi


# milvus_ivfpq collection name
# collection_name=collection_${dataset}_${label_attr}
collection_name=collection_${dataset}_${label_attr}_${partition_size_M}
#_${partition_size_M}
milvus_coll_path=/var/lib/milvus/etcd/data/${collection_name}/   # useless

# milvus_hnsw collection name
hnsw_collection_name=collection_hnsw_${dataset}_${label_attr}_efc${ef_construction}
milvus_hnsw_coll_path=/var/lib/milvus/etcd/data/${hnsw_collection_name}/    # useless



#acorn res path
acorn_root=${root}acorn/
acorn_index_root=${acorn_root}index/
acorn_index_file=${acorn_index_root}index_acorn_${label_attr}_M${M}_ga${gamma}_Mb${M_beta}
# acorn_result_root=${acorn_root}result/


#acorn_rng res path
acorn_rng_root=${root}acorn_rng/
acorn_rng_index_root=${acorn_rng_root}index/
acorn_rng_index_file=${acorn_rng_index_root}index_acorn_rng_${label_attr}_M${M}_ga${gamma}_Mb${M_beta}

#acorn_kgraph res path
acorn_kg_root=${root}acorn_kg/
acorn_kg_index_root=${acorn_kg_root}index/
acorn_kg_index_file=${acorn_kg_index_root}index_acorn_kg_${label_attr}_M${M}_ga${gamma}_Mb${M_beta}

#ivfpq res path
ivfpq_root=${root}ivfpq/
ivfpq_index_root=${ivfpq_root}index/
ivfpq_index_file=${ivfpq_index_root}index_ivfpq_${label_attr}_${partition_size_M}

#hnsw res path
hnsw_root=${root}hnsw/
hnsw_index_root=${hnsw_root}index/
hnsw_index_file=${hnsw_index_root}index_hnsw_${label_attr}_${M}_${ef_construction}

#hnsw kgraph path
hnsw_kg_root=${root}hnsw_kg/
hnsw_kg_index_root=${hnsw_kg_root}index/
hnsw_kg_index_file=${hnsw_kg_index_root}index_hnsw_kg_${label_attr}_${M}_${ef_construction}


#hnsw 2hop path
hnsw_2hop_root=${root}hnsw_2hop/
hnsw_2hop_index_root=${hnsw_2hop_root}index/
hnsw_2hop_index_file=${hnsw_2hop_index_root}index_hnsw_kg_${label_attr}_${M}_${ef_construction}

# airship res path
airship_root=${root}airship/
airship_index_root=${airship_root}index/
airship_index_file=${airship_index_root}index_hnsw_${label_attr}_${M}_${ef_construction}_${alter_ratio}

#DiskAnn need to transform fvecs to bin, and its index/result path

diskann_root=${root}diskann/
diskann_index_root=${diskann_root}index/
diskann_index_label_root=${diskann_index_root}index_diskann_${label_attr}_M${M}_efc${ef_construction}/
diskann_index_file=${diskann_index_label_root}index_diskann

diskann_result_root=${diskann_root}result/
diskann_result_path=${diskann_result_root}result_${label_attr}_efs${ef_search}

keyword_query_range_file=${label_path}sift_qrange_keyword${query_attr}.txt #only key word query supported
label_file=${label_path}data_attr.txt

#DiskAnn need to transform fvecs to bin, and its index/result path

diskann_stit_root=${root}diskann_stitched/
diskann_stit_index_root=${diskann_stit_root}index/
diskann_stit_index_label_root=${diskann_stit_index_root}index_diskann_${label_attr}_M${M}_efc${ef_construction}_stR${Stitched_R}/
diskann_stit_index_file=${diskann_stit_index_label_root}index_diskann

diskann_stit_result_root=${diskann_root}result/
diskann_stit_result_path=${diskann_stit_result_root}result_${label_attr}_efs${ef_search}

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


#rii
rii_root=${root}rii/
rii_index_root=${rii_root}index/
rii_index_file=${rii_index_root}index_rii_${partition_size_M}_${label_attr}

#serf
serf_root=${root}serf/
serf_index_root=${serf_root}serf_index/
serf_index_file=${serf_index_root}index_serf_efmax${ef_max}_M${serf_M}_efs${ef_construction}_${label_attr}

#serf_kg
serf_kg_root=${root}serf_kg/
serf_kg_index_root=${serf_kg_root}serf_kg_index/
serf_kg_index_file=${serf_kg_index_root}index_serf_kg_efmax${ef_max}_M${serf_M}_efs${ef_construction}_${label_attr}

#dsg
dsg_root=${root}dsg/
dsg_index_root=${dsg_root}dsg_index/
dsg_index_file=${dsg_index_root}index_dsg_efmax${ef_max}_efc${ef_construction}_M${serf_M}_${label_attr}

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

#unify
unify_root=${root}unify/
unify_index_root=${unify_root}index/
unify_index_file=${unify_index_root}index_unify_${label_attr}_M${M}_efc${ef_construction}_B${B_unify}
unify_result_root=${unify_root}result/
unify_result_file=${unify_result_root}result_${label_attr}