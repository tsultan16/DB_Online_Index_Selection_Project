"""
    WFIT Driver

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

    # load tpch static workload from a file
    with open('../../../PostgreSQL/ssb_static_workload_2.pkl', 'rb') as f:
        workload_dict = pickle.load(f) 

    workload_metadata = workload_dict['metadata']
    workload = workload_dict['workload']    

    print(f"Loaded static workload from file with {len(workload)} queries.")
    print(f"Num rounds: {workload_metadata['num_rounds']}, Templates per round: {workload_metadata['template_sequence']}")

    # instantiate WFIT
    use_simple_cost = False #True
    max_indexes_per_table = 9 #6 
    max_U = 72  #70
    ibg_max_nodes = 300 #500
    doi_max_nodes = 20 #20
    max_doi_iters_per_node = 100 #200
    normalize_doi = True # False
    idxCnt = 65 # 50
    stateCnt =  400 #400
    rand_cnt = 500 #200
    enable_stable_partition_locking = False
    
    
    if batch_processing in [1, 2]:
        if not use_simple_cost:
            #wfit = WFIT(max_key_columns=3, include_cols=True, max_include_columns=3, simple_cost=False, enable_stable_partition_locking=enable_stable_partition_locking, max_indexes_per_table=max_indexes_per_table, max_U=max_U, ibg_max_nodes=ibg_max_nodes, doi_max_nodes=doi_max_nodes, max_doi_iters_per_node=max_doi_iters_per_node, normalize_doi=normalize_doi, idxCnt=idxCnt, stateCnt=stateCnt, rand_cnt=rand_cnt, execution_cost_scaling=1e-6,creation_cost_fudge_factor=0.5e-3) 

            #wfit = WFIT(max_key_columns=3, include_cols=True, max_include_columns=3, simple_cost=False, enable_stable_partition_locking=enable_stable_partition_locking, max_indexes_per_table=max_indexes_per_table, max_U=max_U, ibg_max_nodes=ibg_max_nodes, doi_max_nodes=doi_max_nodes, max_doi_iters_per_node=max_doi_iters_per_node, normalize_doi=normalize_doi, idxCnt=idxCnt, stateCnt=stateCnt, rand_cnt=rand_cnt, execution_cost_scaling=1e-6,creation_cost_fudge_factor=1e-6) 

            ibg_max_nodes = 150
            
            wfit = WFIT(max_key_columns=3, include_cols=True, max_include_columns=3, simple_cost=False, enable_stable_partition_locking=enable_stable_partition_locking, max_indexes_per_table=max_indexes_per_table, max_U=max_U, ibg_max_nodes=ibg_max_nodes, doi_max_nodes=doi_max_nodes, max_doi_iters_per_node=max_doi_iters_per_node, normalize_doi=normalize_doi, idxCnt=idxCnt, stateCnt=stateCnt, rand_cnt=rand_cnt, execution_cost_scaling=1e-6,creation_cost_fudge_factor=1.15e-3)  #1e-5

        else:       
            #wfit = WFIT(max_key_columns=4, include_cols=True, max_include_columns=3, simple_cost=False, max_indexes_per_table=5, max_U=100, ibg_max_nodes=100, doi_max_nodes=50, max_doi_iters_per_node=50, normalize_doi=False, idxCnt=30, stateCnt=500, rand_cnt=200, execution_cost_scaling=1e-5, creation_cost_fudge_factor=0.0) 
            
            #wfit = WFIT(max_key_columns=3, include_cols=True, max_include_columns=3, simple_cost=True, enable_stable_partition_locking=enable_stable_partition_locking, max_indexes_per_table=max_indexes_per_table, max_U=max_U, ibg_max_nodes=ibg_max_nodes, doi_max_nodes=doi_max_nodes, max_doi_iters_per_node=max_doi_iters_per_node, normalize_doi=normalize_doi, idxCnt=idxCnt, stateCnt=stateCnt, rand_cnt=rand_cnt, execution_cost_scaling=1e-6, creation_cost_fudge_factor=1e-6, join_column_discount=0.7) 

            #wfit = WFIT(max_key_columns=3, include_cols=True, max_include_columns=3, simple_cost=True, enable_stable_partition_locking=enable_stable_partition_locking, max_indexes_per_table=max_indexes_per_table, max_U=max_U, ibg_max_nodes=ibg_max_nodes, doi_max_nodes=doi_max_nodes, max_doi_iters_per_node=max_doi_iters_per_node, normalize_doi=normalize_doi, idxCnt=idxCnt, stateCnt=stateCnt, rand_cnt=rand_cnt, execution_cost_scaling=1e-6, creation_cost_fudge_factor=1e-3, join_column_discount=0.7) 

            wfit = WFIT(max_key_columns=3, include_cols=True, max_include_columns=3, simple_cost=True, enable_stable_partition_locking=enable_stable_partition_locking, max_indexes_per_table=max_indexes_per_table, max_U=max_U, ibg_max_nodes=ibg_max_nodes, doi_max_nodes=doi_max_nodes, max_doi_iters_per_node=max_doi_iters_per_node, normalize_doi=normalize_doi, idxCnt=idxCnt, stateCnt=stateCnt, rand_cnt=rand_cnt, execution_cost_scaling=1e-6, creation_cost_fudge_factor=1e-4, join_column_discount=0.7) 


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
            #wfit.process_WFIT_batch(workload[i], restart_server=True, clear_cache=True, remove_stale_U=True, materialize=False, execute=False, verbose=True)
            print("\n\n")

        # save experiment results to pickle file
        results = {}
        results["batch_recommmendation_time"] = wfit.batch_recommendation_time
        results["batch_materialization_time"] = wfit.batch_materialization_time
        results["batch_execution_time"] = wfit.batch_execution_time
        results["current_configuration"] = wfit.configuration_stats["current_configuration"]
        results["indexes_added"] = wfit.configuration_stats["indexes_added"]
        results["indexes_removed"] = wfit.configuration_stats["indexes_removed"]
        if not use_simple_cost:
            with open(f'wfit_results_whatif.pkl', 'wb') as f:
                pickle.dump(results, f)
        else:
            with open(f'wfit_results.pkl', 'wb') as f:
                pickle.dump(results, f)    

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
        total_time, execution_time = execute_workload_noIndex(workload[:n], drop_indexes=True, restart_server=True, clear_cache=True)

        # split execution time into miniworkload execution time
        batch_execution_time = [execution_time[i:i+12] for i in range(0, len(execution_time), 12)]

        # save experiment results to pickle file
        results = {}
        results["total_time"] = total_time
        results["batch_execution_time"] = batch_execution_time
        results["num_rounds"] = workload_metadata['num_rounds']
        results["template_sequence"] = workload_metadata['template_sequence']
        
        with open(f'noindex_results.pkl', 'wb') as f:
            pickle.dump(results, f)



    print(f"Done!")

if __name__ == "__main__":
    main()








