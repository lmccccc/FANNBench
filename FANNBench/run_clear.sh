source ./vars.sh construction

operation=$1

if [ "$operation" == "irange" ]; then
    rm ${irange_index_file}
    echo "irange index removed"
fi