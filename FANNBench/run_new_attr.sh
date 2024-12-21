
source ./vars.sh query


rm $dataset_attr_file
rm $query_range_file
rm $ground_truth_file
rm $centroid_file
rm $keyword_query_range_file
rm $label_file
rm $ground_truth_bin_file
rm $attr_bin_file
rm $qrange_bin_file
rm $irange_id2od_file
rm -rf $label_path

./run_attr_generator.sh
./run_qrange_generator.sh
./run_groundtruth_generator.sh
