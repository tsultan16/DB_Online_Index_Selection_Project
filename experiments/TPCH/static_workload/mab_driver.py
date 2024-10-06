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


    # load tpch static workload from a file
    with open('../../PostgreSQL/tpch_static_workload_1.pkl', 'rb') as f:
        workload_dict = pickle.load(f) 

    workload_metadata = workload_dict['metadata']
    workload = workload_dict['workload']    

    print(f"Loaded static workload from file with {len(workload)} queries.")
    print(f"Num rounds: {workload_metadata['num_rounds']}, Templates per round: {workload_metadata['template_sequence']}")


    # process the workload
    if mode == 1:
        # split the workload into batches
        batch_size = len(workload_metadata['template_sequence'])
        num_batches = len(workload) // batch_size
        workload = [workload[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
        
        # instantiate MAB
        mab = MAB(alpha=2.0, alpha_decay_rate=0.995, vlambda=0.5, creation_time_reduction_factor=3, config_memory_MB=1024*12, qoi_memory=3, max_indexes_per_table=4, max_index_columns=3, include_cols=True, max_include_columns=3)

        print(f"Batch processing {n} batches of {batch_size} queries each...")
        for i in range(n):
            print('\n------------------------------------------------------------------')
            print(f"Processing batch ({i+1}/{n})...")
            print('------------------------------------------------------------------')
            mab.step_round(workload[i], verbose=True)
            print("\n\n")

        recommendation_time, materialization_time, execution_time = mab.get_total_times()
        print(f"Total recommendation time: {recommendation_time/1000:.5f} s, Total materialization time: {materialization_time/1000:.2f} s, Total execution time: {execution_time/1000:.2f} s")
        print(f"Total time: {(recommendation_time + materialization_time + execution_time)/1000:5f} s")


    elif mode == 0:
        # execute the workload with no indexes
        execute_workload_noIndex(workload[:n], drop_indexes=True, restart_server=True)

    print(f"Done!")

if __name__ == "__main__":
    main()








