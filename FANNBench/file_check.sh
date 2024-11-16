
#check if succicient file exist
if [ ! -f "$dataset_file" ]; then  
    echo "error, $dataset_file does not exist."
    exit 1
fi
if [ ! -f "$query_file" ]; then  
    echo "error, $query_file does not exist."
    exit 1
fi
if [ ! -f "$dataset_attr_file" ]; then  
    echo "error, $dataset_attr_file does not exist."
    exit 1
fi
if [ ! -f "$query_range_file" ]; then  
    echo "error, $query_range_file does not exist."
    exit 1
fi
if [ ! -f "$ground_truth_file" ]; then  
    echo "error, $ground_truth_file does not exist."
    exit 1
fi
if [ ! -f "$train_file" ]; then  
    echo "warning, $train_file does not exist. Make sure you don't need to use it before start."
fi

if [ ! -d "logs" ]; then  
    mkdir logs
fi

