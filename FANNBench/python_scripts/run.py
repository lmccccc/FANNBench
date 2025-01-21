from var import Vars
import sys
import argparse
from run_functions import run
import multi_query
from utils.defination import check_file
# import multiprocessing

# python run.py --algorithm acorn --mode query
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        help="path of attribution file, formated as json",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="construction or query",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="sift1m, sift10m"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        help="label distribution method, random, in_dist or out_dist",
    )
    # parser.add_argument(
    #     "--label_range",
    #     type=int,
    #     default=None,
    #     help=f"candidate label value, [0, label_cnt]",
    # )
    # parser.add_argument(
    #     "--label_cnt",
    #     type=int,
    #     default=1,
    #     help=f"maxinum label size for each vector, 1 for range query",
    # )
    parser.add_argument(
        "--multi_query", action="store_true", help="Query in parallel"
    )
    
    args = parser.parse_args()
    if args.algorithm is None:
        print("NOTE: No algorithm specified, so aborting")
        parser.print_help()
        sys.exit(0)
    if args.mode is None:
        print("NOTE: No mode specified, so aborting")
        parser.print_help()
        sys.exit(0)
    if args.dataset is None:
        print("NOTE: No dataset specified, so aborting")
        parser.print_help()
        sys.exit(0)

    # num_cores = multiprocessing.cpu_count()

    vars = Vars(algorithm=args.algorithm,
                mode=args.mode, 
                dataset=args.dataset, 
                distribution=args.distribution)
    check_file(vars.dataset_path.dataset_file)
    check_file(vars.label_path.dataset_attr_file)
    check_file(vars.label_path.query_range_file)
    check_file(vars.label_path.ground_truth_file)
    
    if args.multi_query: # parallel query
        assert args.mode != "construction"
        process_pool = []
        if args.algorithm == "acorn":
            for ef_search in multi_query.ef_search_list:
                vars = Vars(algorithm=args.algorithm,
                            mode=args.mode, 
                            dataset=args.dataset, 
                            distribution=args.distribution,
                            ef_search=ef_search)
                process = run(vars=vars, blocking=False)
                process_pool.append(process)
        for process in process_pool:      
            stdout, stderr = process.communicate()
            with open(vars.log_file, 'w') as f:
                if stdout:
                    # print(stdout)
                    with open(vars.log_file, 'a') as f:
                        f.write(stdout)  # Log stdout to file
                if stderr:
                    # print(stderr)
                    with open(vars.log_file, 'a') as f:
                        f.write(stderr)  # Log stderr to file


    else:                                # single construction or query
        run(vars=vars, blocking=True)
        

    


