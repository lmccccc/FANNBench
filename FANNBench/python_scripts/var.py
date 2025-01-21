import datetime
import os
import subprocess

# label range   label_cnt   query_label_cnt   query_label
# <500          1           1                 useless                    # For all methods, query random {1} label for each vector has {1} label too.
# <500          >1          1                 0 ~ label_cnt-1            # For keyword query, query {query_label} for vectors which have at most {label_cnt} labels (vector label is in zipf distribution).
# >0            1           >1                useless                    # For range query, query range [random, random+{query_label_cnt}], each vector has {1} label.
# <500          >1          $label_cnt        useless                    # For nhq_kgraph and nhq_nsw, same dimension for label and query label
class Label:
    def __init__(self, distribution=None):
        # lable generation
        self.distribution="random"  # random, in_dist, out_dist
        self.label_range=1         # the possiable lable selected from [0, label_range). keyword query shall less than 500

        # keyword or range for query. Only one of them can >1 at once
        self.label_cnt=1           # Maximal label size for each vector.  1 or $label_range if $labelrange (keyword), not suitable for: 1.acorn,    2.irange,  3.vamanatree,  4.wst 5.nhq_nsw, 6.nhq_kgraph 7.serf 8.unify
        self.query_label_cnt=1     # Range.              if >1 (range query for [x, x+query_label_cnt]), not suitable for: 1.diskann,  2.nhq_nsw, 3.nhq_kgraph
        self.query_label=1         # Queried label, work only for keyword query(label_cnt > 1)

        if distribution is not None:
            self.distribution = distribution

# milvus, hnsw, ivfpq, rii are subset search, which means they are suitable for both range and keyword query
# for batch query at different efs/nprobe/beamsize, see run_multi_query.sh

# var list              construction                                                         search
# acorn                   M M_beta                                                          ef_search
# diskann_memory          M alpha                                                           L
# diskann_stitched        M alpha Stitched_R                                                L
# hnsw                    M ef_construction                                                 ef_search
# irange                  M ef_construction                                                 (M) ef_search
# ivfpq                   partition_size_M                                                  nprobe
# milvus_ivfpq            partition_size_M                                                  nprobe
# nhq_kgraph              kgraph_L iter S R RANGE(alias M) PL(alias ef_cons) B kgraph_M     weight_search L_search(alias ef_search)
# nhq_nsw                 M ef_construction                                                 ef_search
# rii                     partition_size_M
# serf                    serf_M ef_construction                                            ef_search
# vamana_tree             split_factor                                                      beamsize
# wst                     split_factor shift_factor                                         beamsize final_beam_multiply
# UNIFY                   M ef_construction B_unify                                         ef_search AL

# other vars are fixed inside the code
class BaseVars:
    def __init__(self, ef_search=None):
        # hnsw vars
        self.M = 40  # normally 40
        self.ef_construction = 1000
        self.ef_search = 100

        # serf vars
        self.serf_M = 8  # fixed

        # acorn vars
        self.gamma = 12
        self.M_beta = 64

        # DiskAnn typical
        # self.FilteredLbuild = 12
        self.alpha = 1.2  # fixed
        self.L = 400  # "10 20 30 40 50 100" efsearch

        # DiskANN Stitched
        self.Stitched_R = 60

        # rii and pq based
        self.partition_size_M = 64
        self.nprobe = 10

        # RangeFilteredANN, super opt postfiltering
        self.beamsize = 40
        self.split_factor = 2
        self.shift_factor = 0.5
        self.final_beam_multiply = 16

        # nhq kgraph vars
        self.kgraph_L = 100  # <L> is the parameter controlling the graph quality, larger is more accurate but slower, no smaller than K.
        self.iter = 12       # <iter> is the parameter controlling the maximum iteration times, iter usually < 30.
        self.S = 10          # <S> is the parameter controlling the graph quality, larger is more accurate but slower.
        self.R = 300         # <R> is the parameter controlling the graph quality, larger is more accurate but slower.
        self.RANGE = self.M  # <RANGE> controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.
        self.PL = self.ef_construction  # <PL> controls the quality of the NHQ-NPG_kgraph, the larger the better.
        self.B = 0.4         # <B> controls the quality of the NHQ-NPG_kgraph.
        self.kgraph_M = 1    # <M> controls the edge selection of NHQ-NPG_kgraph.

        self.weight_search = 140000

        # UNIFY vars
        self.B_unify = 8
        self.AL = 8  # "[8, 16, 32]"

        self.alter_ratio = 0.5

        # common vars
        self.K = 10
        self.gt_topk = self.K

        # threads
        self.threads = 128

        self.multi = None

        
        if ef_search is not None:
            self.ef_search = ef_search
        self.L_search = self.ef_search


        title = "Paper Dataset dim N query_size label_cnt query_label distribution label_range query_label_cnt K Threads M serf_M nprobe ef_construction ef_search gamma M_beta alpha L partition_size_M beamSize split_factor shift_factor final_beam_multiply kgraph_L iter S R B kgraph_M weight_search Recall QPS selectivity ConstructionTime IndexSize CompsPerQuery File Memory B_unify AL Stitched_R"
        self.title_list = title.split(' ')
        


    # Get the input argument
    def change_mode(self, multi=None, modified_var=None, modified_var2=None):
        if self.mode not in ["construction", "query", "all"]:
            print("Invalid mode. Please use 'construction' or 'query'.")
            exit(1)
        if self.mode == "construction" or self.mode == "all":
            self.threads=128
        elif self.mode == "query":
            self.threads=1
            if multi is None:
                return self
            if not "multi" in multi:
                print("Invalid multi mode. Please use 'multi_hnsw' or 'multi_diskann' or 'multi_ivfpq' or 'multi_kgraph' or 'multi_vtree' or 'multi_wst' or 'multi_unify'.")
                exit(1)
            self.multi="multi"  
            if multi == "multi_hnsw":
                self.ef_search=modified_var
            elif multi == "multi_diskann":
                self.L=modified_var
            elif multi == "multi_ivfpq":
                self.nprobe=modified_var
            elif multi == "multi_kgraph":
                self.ef_search=modified_var
                self.L_search=modified_var2
            elif multi == "multi_vtree":
                self.beamsize=modified_var
            elif multi == "multi_wst":
                self.beamsize=modified_var
                self.final_beam_multiply=modified_var2
            elif multi == "multi_unify":
                self.AL=modified_var
                self.ef_search=modified_var2
        else:
            print("Invalid mode. Please use 'construction' or 'query' or 'all'.")
            exit(1)


# Set thread number based on the mode
class AttrPath:
    def __init__(self, label:Label):
        self.label_attr = f"sel_{label.label_cnt}_{label.label_range}_{label.distribution}"  # to name different label files
        self.query_attr = f"sel_{label.query_label_cnt}_{label.label_cnt}_{label.label_range}_{label.distribution}"  # nothing with query

# dataset vars
class DatasetPath:
    def __init__(self, dataset):
        if dataset == "sift1m":
            # sift1m
            self.dim = 128
            self.N = 1000000
            self.query_size = 10000
            self.train_size = 100000
            self.dataset = "sift1M"
            self.root = "/mnt/data/mocheng/dataset/sift/"
            self.dataset_file = f"{self.root}sift_base.fvecs"
            self.query_file = f"{self.root}sift_query.fvecs"
            self.train_file = f"{self.root}sift_learn.fvecs"
            self.dataset_bin_file = f"{self.root}data_base.bin"  # to be generated
            self.query_bin_file = f"{self.root}data_query.bin"  # to be generated
        elif dataset == "sift10m":
            # sift10m
            self.dim=128
            self.N=10000000
            self.query_size=10000
            self.train_size=1000000
            self.dataset="sift10M"
            self.root="/mnt/data/mocheng/dataset/sift10m/" 
            self.dataset_file=f"{self.root}sift10m.fvecs"
            self.query_file=f"{self.root}sift10m_query.fvecs"
            self.train_file=f"{self.root}sift10m_train.fvecs"
            self.dataset_bin_file=f"{self.root}data_base.bin" # to be generated
            self.query_bin_file=f"{self.root}data_query.bin"   # to be generated
        else:
            # print error
            print("error dataset ", dataset, " not support")
            exit(-1)



# label file, the name of path is defined in label_attr
class LabelPath:
    def __init__(self, datapath:DatasetPath, attrpath:AttrPath, vars:BaseVars):
        self.label_root = f"{datapath.root}label/"
        if not os.path.isdir(self.label_root):
            os.mkdir(self.label_root)
        self.label_path = f"{self.label_root}{attrpath.label_attr}/"
        if not os.path.isdir(self.label_path):
            os.mkdir(self.label_path)

        self.dataset_attr_file = f"{self.label_path}attr_{attrpath.label_attr}.json"
        self.query_range_file = f"{self.label_path}qrange_{attrpath.label_attr}.json"
        self.ground_truth_file = f"{self.label_path}gt_{attrpath.label_attr}_{vars.gt_topk}_.json"
        self.centroid_file = f"{self.label_path}centroid.py"

        self.keyword_query_range_file = f"{datapath.root}sift_qrange_keyword.txt"  # only key word query supported
        self.label_file = f"{datapath.root}data_attr.txt"
        self.ground_truth_bin_file = f"{datapath.root}sift_gt_{vars.gt_topk}.bin"

        self.qrange_bin_file = f"{datapath.root}data_qrange.bin"  # format: left, right, left, right...
        self.attr_bin_file = f"{datapath.root}data_attr.bin"



class AlgorithmFile:
    def __init__(self, vars:BaseVars, datapath:DatasetPath, attrpath: AttrPath):
        # milvus_ivfpq collection name
        self.collection_name = f"collection_{datapath.dataset}_{attrpath.label_attr}"
        self.milvus_coll_path = f"None"  # useless
        
        # milvus_hnsw collection name
        self.hnsw_collection_name = f"collection_hnsw_{datapath.dataset}_{attrpath.label_attr}_efc{vars.ef_construction}"
        self.milvus_hnsw_coll_path = f"None"  # useless

        #acorn res path
        self.acorn_root = f"{datapath.root}acorn/"
        self.acorn_index_root = f"{self.acorn_root}index/"
        self.acorn_index_file = f"{self.acorn_index_root}index_acorn_{attrpath.label_attr}_M{vars.M}_ga{vars.gamma}_Mb{vars.M_beta}"

        #ivfpq res path
        self.ivfpq_root = f"{datapath.root}ivfpq/"
        self.ivfpq_index_root = f"{self.ivfpq_root}index/"
        self.ivfpq_index_file = f"{self.ivfpq_index_root}index_ivfpq_{attrpath.label_attr}_{vars.partition_size_M}"

        #hnsw res path
        self.hnsw_root = f"{datapath.root}hnsw/"
        self.hnsw_index_root = f"{self.hnsw_root}index/"
        self.hnsw_index_file = f"{self.hnsw_index_root}index_hnsw_{attrpath.label_attr}_{vars.M}_{vars.ef_construction}"
        
        #airship res path
        # self.airship_root = f"{datapath.root}airship/"
        # self.airship_index_root = f"{self.airship_root}index/"
        # self.airship_index_file = f"{self.airship_index_root}index_hnsw_{attrpath.label_attr}_{vars.M}_{vars.ef_construction}_{vars.alter_ratio}"
        
        # DiskAnn paths
        self.diskann_root = f"{datapath.root}diskann/"
        self.diskann_index_root = f"{self.diskann_root}index/"
        self.diskann_index_label_root = f"{self.diskann_index_root}index_diskann_{attrpath.label_attr}_M{vars.M}_efc{vars.ef_construction}/"
        self.diskann_index_file = f"{self.diskann_index_label_root}index_diskann"
        self.diskann_result_root = f"{self.diskann_root}result/"
        self.diskann_result_path = f"{self.diskann_result_root}result_{attrpath.label_attr}_efs{vars.ef_search}"

        # DiskAnn Stitched paths
        self.diskann_stit_root = f"{datapath.root}diskann_stitched/"
        self.diskann_stit_index_root = f"{self.diskann_stit_root}index/"
        self.diskann_stit_index_label_root = f"{self.diskann_stit_index_root}index_diskann_stitched_{attrpath.label_attr}_M{vars.M}_efc{vars.ef_construction}/"
        self.diskann_stit_index_file = f"{self.diskann_stit_index_label_root}index_diskann_stitched"

        self.diskann_stit_result_root = f"{self.diskann_stit_root}result/"
        self.diskann_stit_result_path = f"{self.diskann_stit_result_root}result_{attrpath.label_attr}_efs{vars.ef_search}"

        # RangeFilteredANN (WST super optimized postfiltering) paths
        self.rfann_root = f"{datapath.root}rfann/"
        self.rfann_index_root = f"{self.rfann_root}index/"
        self.rfann_index_prefix = f"{self.rfann_index_root}index_{attrpath.label_attr}/"
        self.rfann_result_root = f"{self.rfann_root}result/"
        self.rfann_result_file = f"{self.rfann_result_root}result_{attrpath.label_attr}.csv"

        # RangeFilteredANN (vamana tree) paths
        self.vtree_root = f"{datapath.root}vtree/"
        self.vtree_index_root = f"{self.vtree_root}index/"
        self.vtree_index_prefix = f"{self.vtree_index_root}index_{attrpath.label_attr}/"
        self.vtree_result_root = f"{self.vtree_root}result/"
        self.vtree_result_file = f"{self.vtree_result_root}result_{attrpath.label_attr}.csv"

        # iRangeGraph paths
        self.irange_root = f"{datapath.root}irange/"
        self.irange_index_root = f"{self.irange_root}index/"
        self.irange_id2od_file = f"{self.irange_index_root}data_id2od_{attrpath.label_attr}"
        self.irange_index_file = f"{self.irange_index_root}data_irange_{attrpath.label_attr}"
        self.irange_result_root = f"{self.irange_root}result/"
        self.irange_result_file = f"{self.irange_result_root}result_{attrpath.label_attr}"

        # rii paths
        self.rii_root = f"{datapath.root}rii/"
        self.rii_index_root = f"{self.rii_root}index/"
        self.rii_index_file = f"{self.rii_index_root}index_rii_{vars.partition_size_M}_{attrpath.label_attr}"

        # serf paths
        self.serf_root = f"{datapath.root}serf/"
        self.serf_index_root = f"{self.serf_root}serf_index/"
        self.serf_index_file = f"{self.serf_index_root}index_serf_{attrpath.label_attr}_{vars.serf_M}_{vars.ef_construction}"

        # nhq_nsw paths
        self.nhq_root = f"{datapath.root}nhq/"
        self.nhq_index_root = f"{self.nhq_root}index/"
        self.nhq_index_model_file = f"{self.nhq_index_root}nhq_index_{attrpath.label_attr}_M{vars.M}_efc{vars.ef_construction}"
        self.nhq_index_attr_file = f"{self.nhq_index_root}nhq_attr_{attrpath.label_attr}_M{vars.M}_efc{vars.ef_construction}"

        # nhq_kgraph paths
        self.nhqkg_root = f"{datapath.root}nhqkg/"
        self.nhqkg_index_root = f"{self.nhqkg_root}index/"
        self.nhqkg_index_model_file = f"{self.nhqkg_index_root}nhqkg_index_{attrpath.label_attr}_L{vars.kgraph_L}_iter{vars.iter}S{vars.S}_R{vars.R}_RANGE{vars.RANGE}_PL{vars.PL}_B{vars.B}_M{vars.kgraph_M}"
        self.nhqkg_index_attr_file = f"{self.nhqkg_index_root}nhqkg_attr_{attrpath.label_attr}_L{vars.kgraph_L}_iter{vars.iter}S{vars.S}_R{vars.R}_RANGE{vars.RANGE}_PL{vars.PL}_B{vars.B}_M{vars.kgraph_M}"
        
        # unify paths
        self.unify_root = f"{datapath.root}unify/"
        self.unify_index_root = f"{self.unify_root}index/"
        self.unify_index_file = f"{self.unify_index_root}index_unify_{attrpath.label_attr}_M{vars.M}_efc{vars.ef_construction}_B{vars.B_unify}"
        self.unify_result_root = f"{self.unify_root}result/"
        self.unify_result_file = f"{self.unify_result_root}result_{attrpath.label_attr}"



# Function to log messages
class Log:
    def __init__(self, log_file):
        self.log_file = log_file
        
    
    def log_message(self, message):
        with open(self.log_file, "a") as log:
            log.write(message + "\n")


class Vars:
    def __init__(self, algorithm, mode, dataset, distribution, ef_search=None):
        self.result_file = "exp_results.csv"
        self.algo = algorithm
        self.mode = mode
        self.dataset = dataset
        self.basevars = BaseVars(ef_search=ef_search)
        self.label = Label(distribution=distribution)
        self.attr_path = AttrPath(self.label)
        self.dataset_path = DatasetPath(self.dataset)
        self.label_path = LabelPath(self.dataset_path, self.attr_path, self.basevars)
        self.algorithm_file = AlgorithmFile(self.basevars, self.dataset_path, self.attr_path)

        now = datetime.datetime.now().strftime('%m-%d-%Y')
        self.dir = f"logs/{now}_{self.dataset}_{self.algo}/"
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)

        if self.algo == "acorn":
            self.log_file = f"{self.dir}summary_{self.algo}_{self.dataset}_efs{self.basevars.ef_search}.txt"
        self.log = Log(self.log_file)

        
        print("log file:", self.log_file)
        # Log start time
        self.log.log_message(f"Start time: {subprocess.check_output(['date', '+%H:%M'], env={'TZ': 'America/Los_Angeles'}).decode().strip()}")

        # Log acorn index file
        self.log.log_message(f"acorn index file: {self.algorithm_file.acorn_index_file}")

        

    def multi_query_mode(self, multi=None, modified_var=None, modified_var2=None):
        self.basevars.change_mode(multi, modified_var, modified_var2)
    
    def update_label(self, distribution, label_range, label_cnt):
        self.label.distribution = distribution
        self.label.label_range = label_range
        self.label.label_cnt = label_cnt
        self.attr_path = AttrPath(self.label)
        self.label_path = LabelPath(self.dataset_path, self.attr_path, self.basevars)
        self.algorithm_file = AlgorithmFile(self.basevars, self.dataset_path, self.attr_path)

    
    

def run_command(command:list, vars:Vars, blocking=True):
    print(f"Running {vars.algo} {vars.mode} blocking:{blocking}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if blocking:
        with open(vars.log_file, 'w') as f:
            # Run the command with Popen

            # Block until the subprocess finishes and capture its output
            # Read both stdout and stderr and write them to the file in real-time
            for stdout_line in iter(process.stdout.readline, ""):
                f.write(stdout_line)
                f.flush()  # Immediately write to the file

            for stderr_line in iter(process.stderr.readline, ""):
                f.write(stderr_line)
                f.flush()  # Immediately write to the file

            stdout, stderr = process.communicate()
            # Optionally, handle any final output after the process finishes
            if stdout:
                # print(stdout)
                f.write(stdout)  # Optionally, write remaining stdout to a file
            if stderr:
                # print(stderr)
                f.write(stderr)  # Optionally, write remaining stderr to a file

            print("Task completed!")
        return process.returncode
    else:
        # print("Running in non-blocking mode...")
        # For non-blocking, you can continue doing other tasks while the subprocess is running
        process.poll()  # Check if the process has finished without blocking
        # print("Subprocess is running asynchronously...")
        return process