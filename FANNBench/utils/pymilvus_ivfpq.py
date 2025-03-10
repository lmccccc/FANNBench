
import sys
from pymilvus import DataType, MilvusClient, utility
import sys
from defination import check_dir, check_file, ivecs_read, fvecs_read, bvecs_read, read_attr, read_file
import numpy as np
import math
import time
import math

# read args
if __name__ == "__main__":
    # ${dataset} 
    # ${N} 
    # ${query_size} 
    # ${dataset_file} 
    # ${query_file} 
    # ${dataset_attr_file} 
    # ${query_range_file} 
    # ${K}
    if not len(sys.argv) == 14 :
        print("error wrong argument size")
        exit()
    else:
        dataset = sys.argv[1]
        print("dataset:", dataset)

        N = int(sys.argv[2])
        print("N:", N)
        
        dataset_file = sys.argv[3]
        print("dataset file:", dataset_file)
        check_file(dataset_file)

        query_file = sys.argv[4]
        print("query file:", query_file)
        check_file(query_file)

        dataset_attr_file = sys.argv[5]
        print("attr file:", dataset_attr_file)
        check_file(dataset_attr_file)

        query_range_file = sys.argv[6]
        print("query range file:", query_range_file)
        check_file(query_range_file)

        groundtruth_file = sys.argv[7]
        print("ground truth file:", groundtruth_file)
        check_file(groundtruth_file)

        K = int(sys.argv[8])
        print("K:", K)
        
        nprobe = int(sys.argv[9])
        print("nprobe:", nprobe)

        c_name = sys.argv[10]
        print("collection name:", c_name)

        mode = sys.argv[11]

        M = int(sys.argv[12])

        d = int(sys.argv[13])


    client = MilvusClient(
        uri="http://localhost:19530"
    )

    # create collection
    if client.has_collection(c_name) and  "query" in mode:
        print("collection ", c_name, " exists")

        client.load_collection(collection_name=c_name, 
                               replica_number=1,
                               load_fields=["id", "vector", "label"])
        res = client.query(
            collection_name=c_name,
            output_fields=["count(*)"]
        )

        print("collection size:", res)

    elif (not client.has_collection(c_name)) and  "query" in mode:
        print("error no such cllection ", c_name)
        sys.exit(-1)
        
        # client.drop_collection(c_name)
    if mode == "drop":
        if (client.has_collection(c_name)):
            print("collection ", c_name, " exists, then drop it")
            client.drop_collection(collection_name=c_name)
        else:
            print("collection ", c_name, " not exist")
        sys.exit(0)
    else:
    # if True:
        if client.has_collection(c_name): # construction mode
            print("collection ", c_name, " exists")
            exit()
            # print("collection ", c_name, " exists, then drop it")
            # client.drop_collection(collection_name=c_name)

        # load data
        dataset = read_file(dataset_file)
        N, _d = dataset.shape
        assert(_d == d)
        print("get dataset size:", N, " d:", _d)
        print("dim:", d)

        attr = read_attr(dataset_attr_file)
        _N = len(attr)
        print("_N:", _N, " N:", N)
        assert(_N == N)

        label_set = set(attr)
        label_cnt = len(label_set)
        partition_size = min(64, label_cnt)

        # create collection
        schema = MilvusClient.create_schema(
            auto_id=False,
            partition_key_field="label",      # default partition=64
            num_partitions=partition_size
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=d)
        schema.add_field(field_name="label", datatype=DataType.INT64)
        
        nlist = int(math.sqrt(N))
        print("partition M:", M)
        print("nlist:", nlist)

        client.create_collection(collection_name=c_name, 
                                    schema=schema,
                                    enable_dynamic_field=True,
                                    consistency_level="Strong")


        
        data = [
            {"id": i, "vector": dataset[i].tolist(), "label": attr[i]}
            for i in range(N)
        ]

        # insert data too large in a batch will cause timeout
        print("start insertion")
        # set start time
        t0 = time.time()
        max_message_size = 67108864
        max_batch_size = max_message_size // (d * 4)
        batch_size = max_batch_size // 2  # Divide by 2 to be safe
        print("batch size:", batch_size, " total batch:", round(N/batch_size))
        for i in range(0, N, batch_size):
            client.insert(collection_name=c_name, data=data[i:min(i+batch_size, N)])
            print("insert batch:", i/batch_size, " of ", round(N/batch_size), " from ", i, " to ", min(i+batch_size, N))

        # client.insert(collection_name=c_name, data=data)
        # print("insert res:", res)
        
        t2 = time.time()
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector", 
            index_type="IVF_PQ",# IVF_FLAT IVF_PQ IVF_SQ8 HNSW SCANN
            metric_type="L2",
            params={ "nlist": nlist, "m": M} # see https://milvus.io/docs/configure_querynode.md#queryNodesegcoreinterimIndexnlist
        )
        client.create_index(
            collection_name=c_name,
            index_params=index_params,
            sync=True # Whether to wait for index creation to complete before returning. Defaults to True.
        )
        t1 = time.time()
        
        client.load_collection(collection_name=c_name, 
                               replica_number=1,
                               load_fields=["id", "vector", "label"])

        res = client.query(
            collection_name=c_name,
            output_fields=["count(*)"]
        )

        print("collection size:", res)

        print("insert suc, time cost:", t1-t2)
        
        if mode == "construction":
            print("construction done")
            exit()


    query = read_file(query_file)
    Nq, _d = query.shape
    print("query d:", _d)
    print("get query szie:", Nq)  
    
    qrange = read_attr(query_range_file)
    qrange = qrange.reshape(-1, 2)
    _Nq = len(qrange)
    print("get range cnt:", _Nq)
    assert(_Nq == Nq)

    gt = read_attr(groundtruth_file)
    gt = gt.reshape(-1, K)
    Ngt = gt.shape[0]
    
    print("ngt=", Ngt, " nq=", Nq, " k=", K)
    assert(Ngt == Nq)

    ids = []
    q_t = 0
    s_params = {"metric_type": "L2", "params": {"nprobe": nprobe}}
    hist = np.array([0 for _ in range(11)], dtype='float')
    total_size = Nq * K
    positive_size = 0
    for i in range(Nq):
        # if(i % 100 == 0):
        #     print("batch ", i)
        q_cnt = 0
        exp = str(qrange[i][0]) + " <= label <= " + str(qrange[i][1])
        # qr = [i for i in range(qrange[i][0], qrange[i][1] + 1)]
        # exp = "label in " + str(qr)
        t0 = time.time()
        res = client.search(
                            collection_name=c_name,
                            data=[query[i].tolist()], 
                            filter=exp,
                            limit=K,
                            search_params=s_params, 
                            group_strict_size=True,
                            )
        t1 = time.time()
        q_t += t1 - t0
        res_id = [x['id'] for x in res[0]]
        ids.append(res_id)
        if(len(res_id) != K):
            print("error, get ", len(res_id), " results, expect ", K)
            print("res:", res)
            print("exp:", exp)

        for top in range(K):
            for gtind in range(K):
                if(ids[i][top] == gt[i][gtind]):
                    positive_size += 1
                    q_cnt += 1
                    break
        q_recall = q_cnt * 1.0 / K
        q_bucket = math.floor(q_recall * 10)
        hist[q_bucket] += 1
        # if i % 100 == 0:
        #     print( "checkpoint ", i, "temp recall:", positive_size * 1.0 / ((i+1)* K))
        #     _hist = hist / ((i+1)*100)
        #     print("temp recall dist(10\% per bucket): ", _hist)

    hist = np.array([0 for _ in range(11)], dtype='float')
    total_size = Nq * K
    positive_size = 0
    # client.drop_collection(collection_name=c_name)
    print("ids shape:", len(ids), len(ids[0]))
    print("gt shape:", gt.shape)
    for qind in range(Nq):
        q_cnt = 0
        for top in range(K):
            for gtind in range(K):
                if(ids[qind][top] == gt[qind][gtind]):
                    positive_size += 1
                    q_cnt += 1
                    break
        q_recall = q_cnt * 1.0 / K
        q_bucket = math.floor(q_recall * 10)
        hist[q_bucket] += 1


    # Get component status
    # status = utility.get_server_status()
    # print("Milvus Status:", status)

    # # Get system information
    # metrics = utility.get_system_metrics()
    # print("System Metrics:", metrics)

    recall = positive_size * 1.0 / total_size
    print("milvus get ", positive_size, " postive res from ", total_size, " results, recall@", K, ":", recall)
    hist = hist / Nq
    print("recall dist(10\% per bucket): ", hist)
    print(Nq, " queries cost:", q_t, ", qps:" , Nq / q_t)


        