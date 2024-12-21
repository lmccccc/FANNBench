# plot attr distribution
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
sys.path.append("utils") 
from defination import read_attr, read_keywords
if __name__ == "__main__":
    dataset = sys.argv[1]
    query_label_cnt = int(sys.argv[2])
    label_range = int(sys.argv[3])
    label_cnt = int(sys.argv[4])
    query_label = int(sys.argv[5])
    distribution = sys.argv[6]
    dataset_attr_file = sys.argv[7]
    query_range_file = sys.argv[8]
    dataset_size = int(sys.argv[9])
    query_size = int(sys.argv[10])

    if label_cnt == 1:
        attr = read_attr(dataset_attr_file)
        plot_attr = attr
    else:
        plot_attr = []
        attr = read_keywords(dataset_attr_file)
        for i in range(dataset_size):
            plot_attr.extend(attr[i])
    
    print("plot query attr size:", len(plot_attr))
    bucket_cnt = label_range // 10
    bucket_width = math.ceil(label_range / bucket_cnt)
    bins = np.arange(0, label_range, bucket_width)

    plt.figure(figsize=(10, 6))
    plt.hist(plot_attr, bins=bins, alpha=0.7)
    plt.xlabel('Attribute Value')
    plt.ylabel('Count')
    plt.title('Attribute Distribution')

    bin_edges = bins[:-1] + 0.5  # Calculate midpoints between bars
    # Set x-ticks to every 5th bin midpoint
    skip_xtick = bucket_cnt // 10
    tick_positions = bin_edges[::skip_xtick]
    plt.xticks(tick_positions, labels=[str(int(b)) for b in bins[:-1][::skip_xtick]], rotation=45)
    plt.show()
    # save plotW
    plot_file = "plot/dist_" + dataset_attr_file.split('.')[0].split('/')[-1] + ".png"
    print("save plot to file:", plot_file)
    plt.savefig(plot_file)



    # query dist plot
    query = read_attr(query_range_file)
    query = np.array(query)
    query_center = ((query[::2] + query[1::2]) / 2).tolist()

    print("plot attr size:", len(query_center))
    bucket_cnt = label_range // 100
    bucket_width = math.ceil(label_range / bucket_cnt)
    bins = np.arange(0, label_range, bucket_width)

    plt.figure(figsize=(10, 6))
    plt.hist(query_center, bins=bins, alpha=0.7)
    plt.xlabel('Attribute Value')
    plt.ylabel('Count')
    plt.title('Queyr Attribute Distribution')

    bin_edges = bins[:-1] + 0.5  # Calculate midpoints between bars
    # Set x-ticks to every 5th bin midpoint
    skip_xtick = bucket_cnt // 10
    tick_positions = bin_edges[::skip_xtick]
    plt.xticks(tick_positions, labels=[str(int(b)) for b in bins[:-1][::skip_xtick]], rotation=45)
    plt.show()
    # save plotW
    plot_file = "plot/dist_query_" + dataset_attr_file.split('.')[0].split('/')[-1] + ".png"
    print("save plot to file:", plot_file)
    plt.savefig(plot_file)