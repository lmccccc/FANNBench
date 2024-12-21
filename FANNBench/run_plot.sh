source ./vars.sh query # anything is fine since no construction or query to do

mkdir plot

if [ "$1" == "construction" ]; then
    python utils/plot/construction_time_and_size.py exp_results.csv $dataset
elif [ "$1" == "query" ]; then
    python utils/plot/recall_qps.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution 
elif [ "$1" == "dist" ]; then
    python utils/plot/attr_dist.py  $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $dataset_attr_file \
                                    $query_range_file \
                                    $N \
                                    $query_size

fi
# echo "plot done"

