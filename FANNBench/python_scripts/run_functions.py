from var import Vars
import run_acorn
from utils.extract_results_new import extract_results

def run(vars:Vars, blocking:bool=True):
    returncode = -1
    if vars.algo == "acorn":
        returncode = run_acorn.run(vars, blocking)
    
    if returncode == 0:
        extract_results(vars)
    else:
        vars.log.log_message("Error in run function, return code {}".format(returncode))
        print("Error in run function")
