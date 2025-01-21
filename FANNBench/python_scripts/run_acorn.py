import os
from var import Vars, run_command
from utils.defination import check_file
# Get the current date and format it as day-month-year
def run(vars:Vars, blocking:bool=True):
    # Construction mode
    if vars.mode in ["construction", "all"]:
        if os.path.exists(vars.algorithm_file.acorn_index_file):
            vars.log.log_message("acorn index file already exists")
        else:
            vars.log.log_message("construct index")
            command = [
                "/bin/time", "-v", "-p", "../ACORN/build/demos/acorn_build",
                vars.dataset_path.dataset,
                str(vars.dataset_path.N),
                str(vars.basevars.gamma),
                str(vars.basevars.M),
                str(vars.basevars.M_beta),
                str(vars.basevars.K),
                str(vars.basevars.threads),
                vars.dataset_path.dataset_file,
                vars.label_path.dataset_attr_file,
                vars.algorithm_file.acorn_index_file,
                str(vars.dataset_path.dim)
            ]
            # result = subprocess.run(command, capture_output=True, text=True)
            returncode = run_command(command, vars, blocking)
            if returncode != 0:
                vars.log.log_message(f"acorn index failed with exit status {returncode}")
            else:
                vars.log.log_message("acorn index ran successfully")
        return returncode

    # Query mode
    if vars.mode in ["query", "all"]:
        check_file(vars.algorithm_file.acorn_index_file)
        if vars.label.label_cnt > 1:
            vars.log.log_message(f"start query, efs: {vars.basevars.ef_search}")
            command = [
                "/bin/time", "-v", "-p", "../ACORN/build/demos/acorn_query_keyword",
                vars.dataset_path.dataset,
                str(vars.dataset_path.N),
                str(vars.basevars.gamma),
                str(vars.basevars.M),
                str(vars.basevars.M_beta),
                str(vars.basevars.K),
                str(vars.basevars.threads),
                vars.dataset_path.dataset_file,
                vars.label_path.dataset_attr_file,
                vars.algorithm_file.acorn_index_file,
                str(vars.dataset_path.dim),
                str(vars.basevars.ef_search)
            ]
            process = run_command(command, vars, blocking)
            return process
        else:
            print("Error: label_cnt must be greater than 1")
            exit(-1)
    