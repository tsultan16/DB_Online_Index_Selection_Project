"""
    MAB Driver

"""

import sys
import pickle 
from mab_v2 import *


def main():
    # Check if there are enough arguments
    if len(sys.argv) < 2:
        print("Usage: python mab_driver.py <mode> <num_rounds/num_queries>")
        sys.exit(1)

    # Access command line arguments
    script_name = sys.argv[0]
    arguments = sys.argv[1:]  # All arguments except the script name
    #print(f"Script name: {script_name}")
    #print("Arguments:", arguments)
    mode = int(arguments[0])
    n = int(arguments[1])

    if mode not in [0, 1]:
        print("Invalid mode option. Exiting...")
        sys.exit(1)


    # load tpch dynamic workload from a file
    with open('../../../PostgreSQL/tpch_dynamic_shifting_workload_1.pkl', 'rb') as f:
        workload_dict = pickle.load(f) 

    workload_metadata = workload_dict['metadata']
    workload = workload_dict['workload']    

    print(f"Loaded dynamic shifting workload from file with {len(workload)} queries.")
    print(f"Num rounds: {workload_metadata['num_rounds']}, Templates in sequence 1: {workload_metadata['sequence_1']}, Templates in sequence 2: {workload_metadata['sequence_2']}")


    # process the workload
    if mode == 1:
        # instantiate MAB
        mab = MAB(alpha=20.0, alpha_decay_rate=0.995, vlambda=0.2, creation_time_reduction_factor=3, config_memory_MB=1024*20, qoi_memory=3, max_indexes_per_table=9, max_index_columns=3, include_cols=True, max_include_columns=3)

        print(f"Batch processing {n} batches of {5} queries each...")
        for i in range(n):
            print('\n------------------------------------------------------------------')
            print(f"Processing batch ({i+1}/{n})...")
            print('------------------------------------------------------------------')
            mab.step_round(workload[i], verbose=True)
            print("\n\n")

        recommendation_time, materialization_time, execution_time = mab.get_total_times()
        print(f"Total recommendation time: {recommendation_time/1000:.5f} s, Total materialization time: {materialization_time/1000:.2f} s, Total execution time: {execution_time/1000:.2f} s")
        print(f"Total time: {(recommendation_time + materialization_time + execution_time)/1000:5f} s")

        # save experiment results to pickle file
        results = {}
        results["batch_recommmendation_time"] = mab.recommendation_time
        results["batch_materialization_time"] = mab.materialization_time
        results["batch_execution_time"] = mab.execution_time
        results["current_configuration"] = mab.current_donfiguration
        results["indexes_added"] = mab.indexes_added
        results["indexes_removed"] = mab.indexes_removed
        with open(f'mab_results.pkl', 'wb') as f:
            pickle.dump(results, f)

    elif mode == 0:
        # execute the workload with no indexes
        execute_workload_noIndex(workload[:n], drop_indexes=True, restart_server=True)

    print(f"Done!")

if __name__ == "__main__":
    main()








