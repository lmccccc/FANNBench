source ./vars.sh query # anything is fine since no construction or query to do

mkdir plot
# distribution="random"
# label_range=5
# label_attr=sel_${label_cnt}_${label_range}_${distribution} # to name different label files
# query_attr=sel_${query_label_cnt}_${label_cnt}_${label_range}_${distribution} # nothing with query

if [ "$1" == "construct" ]; then
    python utils/plot/construction_time_and_size.py exp_results.csv $dataset $label_attr
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
elif [ "$1" == "queryplot" ]; then
    python utils/plot/recall_qps2.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "cpqqps" ]; then
    python utils/plot/cpq_qps.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "recallqps" ]; then
    python utils/plot/recall_qps.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "recallqpsserf" ]; then
    python utils/plot/recall_qps_serf.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "recallqpsoneth" ]; then
    python utils/plot/recall_qps_oneth.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "recallqpsstitched" ]; then
    python utils/plot/recall_qps_stitched.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "recallqpslabel" ]; then
    python utils/plot/recall_qps_label.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "selqpsbest" ]; then
    python utils/plot/sel_qps_best.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "entrycmp" ]; then
    python utils/plot/entry_cmp.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "entrysizecmp" ]; then
    python utils/plot/entry_size_cmp.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "prunecmp" ]; then
    python utils/plot/prune_cmp.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "selcpqfilter" ]; then
    python utils/plot/sel_cpq_filtering.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "selqpsbestlabel" ]; then
    python utils/plot/sel_qps_best_label.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "recallqpsdist" ]; then
    python utils/plot/recall_qps_dist.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "qpsbar" ]; then
    python utils/plot/recall_qps3.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "qpsparam" ]; then
    python utils/plot/qps_para.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "qpsparamlabel" ]; then
    python utils/plot/qps_para_label.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "qpsbarlabel" ]; then
    python utils/plot/recall_bar_label.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "queryplotkeyword" ]; then
    python utils/plot/recall_qps_keyword.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr
elif [ "$1" == "index" ]; then
    python utils/plot/index.py exp_results.csv \
                                    $dataset \
                                    $query_label_cnt \
                                    $label_range \
                                    $label_cnt \
                                    $query_label \
                                    $distribution \
                                    $label_attr \
                                    $query_attr

fi
# echo "plot done"