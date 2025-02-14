import sys
from defination import check_file, check_dir, fvecs_read, read_attr
import json
import numpy as np

if __name__ == "__main__":

    data_file = sys.argv[1]
    print("data file:", data_file)
    check_file(data_file)

    attr_file = sys.argv[2]
    check_file(attr_file)
    print("attr file:", attr_file)
    
    query_file = sys.argv[3]
    print("query file:", query_file)
    check_file(query_file)

    qrange_file = sys.argv[4]
    print("query range file:", qrange_file)
    check_file(qrange_file)

    gt_file = sys.argv[5]
    print("ground truth file:", gt_file)
    check_file(gt_file)

    k = int(sys.argv[6])
    print("k:", k)


    data = fvecs_read(data_file)
    query = fvecs_read(query_file)
    qrange = read_attr(qrange_file)
    attr = read_attr(attr_file)
    gt = read_attr(gt_file)

    sample_data_size2 = 100
    sampled_index2 = np.random.choice(data.shape[0], sample_data_size2, replace=False)
    sampled_data2 = data[sampled_index2]
    print("sampled data shape:", sampled_data2.shape)
    
    dis = []
    for j in sampled_data2:
        one_dis_list = np.linalg.norm(data - j, axis=1)
        # print("one_dis_list shape:", one_dis_list.shape)
        top_k_dis = np.sort(one_dis_list)[:k]
        _avg_dis = np.mean(top_k_dis)
        dis.append(_avg_dis)
    dis = np.array(dis)
    avg_dis = np.mean(dis)
    print("avg_dis:", avg_dis)

    # dis for queried subset
    sampled_query_index = np.random.choice(query.shape[0], sample_data_size2, replace=False)
    print("sampled query index shape:", sampled_query_index.shape)
    # query = query[sampled_query_index]
    # qrange = qrange[sampled_query_index]

    dis = []
    for ind in sampled_query_index:
        condition = (attr>=qrange[ind*2]) & (attr<= qrange[ind*2+1])
        subset_index = np.where(condition)[0]
        subset = data[subset_index]

        sample_subset_size = 10
        sample_subset_index = np.random.choice(subset.shape[0], min(sample_subset_size, subset.shape[0]), replace=False)
        for ind2 in sample_subset_index:
            one_dis_list = np.linalg.norm(subset - subset[ind2],axis=1)
            top_k_dis = np.sort(one_dis_list)[:k]
            _avg_dis = np.mean(top_k_dis)
            dis.append(_avg_dis)
    dis = np.array(dis)
    sub_avg_dis = np.mean(dis)
    print("avg_dis for subset:", sub_avg_dis)
    s_dense = sub_avg_dis / avg_dis
    print("subset density:", s_dense)


    # dis for query to groundtruth
    dis = []
    for ind in sampled_query_index:
        kgt = gt[ind*k:(ind+1)*k]
        one_dis_list = np.linalg.norm(data[kgt] - query[ind])
        _avg_dis = np.mean(one_dis_list)
        dis.append(_avg_dis)
    dis = np.array(dis)
    gt_avg_dis = np.mean(dis)
    print("avg_dis for query to groundtruth:", gt_avg_dis)
    gt_dense = gt_avg_dis / avg_dis
    print("groundtruth density:", gt_dense)


            