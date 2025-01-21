source ./vars.sh query # anything is fine since no construction or query to do

mkdir plot
# distribution="random"
# label_range=5
# label_attr=sel_${label_cnt}_${label_range}_${distribution} # to name different label files
# query_attr=sel_${query_label_cnt}_${label_cnt}_${label_range}_${distribution} # nothing with query

if [ "$1" == "construct" ]; then
    python utils/plot/construction_time_and_size.py exp_results.csv $dataset $label_attr
elif [ "$1" == "query" ]; then
    python utils/plot/recall_qps.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
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
                                    $query_size \
                                    $label_attr \
                                    $query_attr

fi
# echo "plot done"

