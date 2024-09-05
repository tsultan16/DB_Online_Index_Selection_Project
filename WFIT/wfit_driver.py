"""
    WFIT Driver

    TODO: add command line arguments to specify the workload type, number of rounds, etc.
"""

import sys
import pickle 
from wfit import *



def main():
    # Check if there are enough arguments
    if len(sys.argv) < 2:
        print("Usage: python wfit_driver.py <num_queries> ...")
        sys.exit(1)

    # Access command line arguments
    script_name = sys.argv[0]
    arguments = sys.argv[1:]  # All arguments except the script name
    #print(f"Script name: {script_name}")
    #print("Arguments:", arguments)
    batch_processing = int(arguments[0])
    n = int(arguments[1])

    if batch_processing not in [0, 1, 2]:
        print("Invalid batch processing flag. Exiting...")
        sys.exit(1)

    # load workload from a file
    with open('ssb_static_workload_4.pkl', 'rb') as f:
        workload_dict = pickle.load(f) 

    workload_metadata = workload_dict['metadata']
    workload = workload_dict['workload']    

    print(f"Loaded static workload from file with {len(workload)} queries.")
    print(f"Num rounds: {workload_metadata['num_rounds']}")
    print(f"Template sequence: {workload_metadata['template_sequence']}")

    # instantiate WFIT
    #wfit = WFIT(S_0, max_key_columns=3, include_cols=False, max_indexes_per_table=3, max_U=None, ibg_max_nodes=100, doi_max_nodes=50, max_doi_iters_per_node=200, normalize_doi=False, idxCnt=50, stateCnt=500, rand_cnt=100, execution_cost_scaling=1e-6,creation_cost_fudge_factor=1e-4) 
    
    #wfit = WFIT(S_0, max_key_columns=3, include_cols=False, max_indexes_per_table=5, max_U=None, ibg_max_nodes=100, doi_max_nodes=50, max_doi_iters_per_node=200, normalize_doi=False, idxCnt=30, stateCnt=500, rand_cnt=200, execution_cost_scaling=1e-6,creation_cost_fudge_factor=2.5e-3) 
    
    if batch_processing in [1, 2]:
        #wfit = WFIT(max_key_columns=3, include_cols=True, max_indexes_per_table=5, max_U=100, ibg_max_nodes=100, doi_max_nodes=50, max_doi_iters_per_node=200, normalize_doi=False, idxCnt=30, stateCnt=500, rand_cnt=200, execution_cost_scaling=1e-6,creation_cost_fudge_factor=2e-3) 
        wfit = WFIT(max_key_columns=3, include_cols=True, max_indexes_per_table=5, max_U=100, ibg_max_nodes=100, doi_max_nodes=50, max_doi_iters_per_node=200, normalize_doi=False, idxCnt=30, stateCnt=500, rand_cnt=200, execution_cost_scaling=1e-6,creation_cost_fudge_factor=1.25e-3) 


    # process the workload
    if batch_processing == 1:
        # split the workload into batches
        batch_size = len(workload_metadata['template_sequence'])
        num_batches = len(workload) // batch_size
        workload = [workload[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
        
        print(f"Batch processing {n} batches of {batch_size} queries each...")
        for i in range(n):
            print('------------------------------------------------------------------')
            print(f"Processing batch ({i+1}/{n})...")
            print('------------------------------------------------------------------')
            wfit.process_WFIT_batch(workload[i], restart_server=True, clear_cache=True, remove_stale_U=True, materialize=True, execute=True, verbose=True)
            print("\n\n")

    elif batch_processing == 2:    
        print(f"Processing {n} queries from the workload...\n\n")
        for i, query in enumerate(workload[:n]):
            print('------------------------------------------------------------------')
            print(f"Processing query {i+1}")
            print('------------------------------------------------------------------')
            wfit.process_WFIT(query, remove_stale_U=False, remove_stale_freq=1, execute=True, materialize=True, verbose=True)
            print("\n\n")

    elif batch_processing == 0:
        # execute the workload with no indexes
        execute_workload_noIndex(workload[:n], drop_indexes=True, restart_server=True, clear_cache=True)

    print(f"Done!")

if __name__ == "__main__":
    main()








