"""
    MAB Implementation (v2)
"""


import sys
import os

# Add Postgres directory to sys.path
module_path = os.path.abspath('/home/tanzid/Code/DBMS/PostgreSQL')
if module_path not in sys.path:
    sys.path.append(module_path)

from pg_utils import *
from ssb_qgen_class import *

import numpy as np
from collections import defaultdict
from functools import lru_cache
import time



# get table size, row count, sequential scan time and column info
@lru_cache(maxsize=None)
def get_table_and_column_details():
    ssb_schema, pk_columns = get_ssb_schema()
    columns = {}
    tables = {}
    for table_name in ssb_schema:
        conn = create_connection()
        table_info = get_table_size_and_row_count(conn, table_name)
        table_scan_time = get_sequential_scan_time(conn, table_name)
        #print(f"Table name : {table_name}, table info : {table_info}")
        tables[table_name] =  {**table_info, **{"pk_columns": pk_columns[table_name], "sequential_scan_time": [table_scan_time]}}
        close_connection(conn)
        columns[table_name] = [c[0] for c in ssb_schema[table_name]]  
         
    return tables, columns



class MAB:
    database_size_cache = None
    table_info_cache = None
    column_info_cache = None

    def __init__(self, alpha=1.0, vlambda=0.5, alpha_decay_rate=1.0, creation_time_reduction_factor=1, config_memory_MB=1024, qoi_memory=4, max_indexes_per_table=5, max_index_columns=3, include_cols=False):
        # define Lin UCB parameters
        self.alpha = alpha     # UCB exploration parameter
        self.vlambda = vlambda # regularization parameter
        self.alpha_decay_rate = alpha_decay_rate  # decay rate for alpha
        self.creation_time_reduction_factor = creation_time_reduction_factor  # factor to reduce the impact of creation cost on estimated expected reward
        self.config_memory_MB = config_memory_MB  # memory budget for storing indexes
        self.qoi_memory = qoi_memory  # how far back to look for queries of interest (QoIs)

        # get database size
        if MAB.database_size_cache is None:
            conn = create_connection()
            MAB.database_size_cache = get_database_size(conn)
            close_connection(conn)
        else:
            print("Found database size in cache")    
        self.database_size = MAB.database_size_cache    
        print(f"Database size: {self.database_size}")
        # get all tables and columns info
        if MAB.table_info_cache is None:
            MAB.table_info_cache, MAB.column_info_cache = get_table_and_column_details()
        else:
            print("Found table and column info in cache")    
        self.tables = MAB.table_info_cache
        self.all_columns = MAB.column_info_cache    
        self.num_columns = sum([len(columns) for columns in self.all_columns.values()])
        print(f"Table info: {self.tables}")
        print(f"Total number of columns: {self.num_columns}")

        # drop all existing indexes
        print("Dropping all existing secondary indexes...")
        conn = create_connection()
        drop_all_indexes(conn)
        close_connection(conn)

        # context vector dims  
        self.context_size = self.num_columns + self.num_columns + 2  # index_columns + include_columns + derived_context
        print(f"Context vector size: {self.context_size}")

        # create a mapping from column name to integer 
        self.columns_to_idx = {}
        i = 0
        for table_name, columns in self.all_columns.items():
            for column in columns:
                self.columns_to_idx[column] = i
                i += 1

        self.idx_to_columns = {v: k for k, v in self.columns_to_idx.items()}   

        # initialize matrix V and vector b
        self.V = np.identity(self.context_size) * self.vlambda
        self.b = np.zeros(shape=(self.context_size, 1))
        self.context_vectors = None
        self.upper_bounds  = None
        self.index_selection_count = defaultdict(int)
        self.query_store = {}
        self.selected_indices_last_round = {}
        self.table_scan_times = defaultdict(list)
        self.index_average_reward = defaultdict(float)
    
        # initialize query store
        self.query_store = {}
        # track current round    
        self.current_round = 0
        # create a cache for column context vectors
        self.column_context_cache = {}
        # cache for storing index stats
        self.index_average_reward = defaultdict(float)
        self.index_creation_time = defaultdict(list)
        self.index_size = {}
        self.index_usage_rounds = defaultdict(list) # tracks which rounds an index was used in and the scan time
        self.index_selected_rounds = defaultdict(list) # tracks which rounds an index was selected in
        self.table_scan_times = {}
        for table_name in self.tables.keys():
            self.table_scan_times[table_name] = self.tables[table_name]["sequential_scan_time"]
        # keep copy of indexes selected in previous round and current round candidates
        self.selected_indexes_last_round = {}
        self.candidate_indexes = {}
 
        # track materialization time of indexes for current round
        self.current_round_index_creation_time = {}

        # track recommendation time, config materialization time and miniworkload execution time
        self.recommendation_time = []
        self.materialization_time = []
        self.execution_time = []

        # constants
        self.MAX_INDEXES_PER_TABLE = max_indexes_per_table
        self.MAX_INDEX_COLUMNS = max_index_columns
        self.INCLUDE_COLS = include_cols
        self.SMALL_TABLE_IGNORE = 1000
        self.TABLE_MIN_SELECTIVITY = 0.5
        self.MAX_INDEXES_PER_TABLE = 5
        self.TABLE_SCAN_TIME_HISTORY = 1000

        
    # step through a round of the MAB
    def step_round(self, mini_workload, restart_server=True, verbose=False):
        self.current_round += 1
        print(f"\nRunning MAB for round: {self.current_round}")

        start_time = time.perf_counter()

        # identify new query templates from the mini workload and update stats    
        if verbose: print(f"Identifying new query templates from the mini workload...")
        self.identify_new_query_templates(mini_workload, verbose)

        # select queries of interest (QoIs) from past workload and use them to extract candidate indexes, i.e. bandit arms
        if verbose: print(f"Selecting QoIs and extracting candidate indexes...")
        QoIs = self.select_queries_of_interest(mini_workload, verbose)
        self.candidate_indexes = self.extract_candidate_indexes(QoIs, verbose)

        if len(self.candidate_indexes) == 0:
            print(f"No candidate indexes extracted. Skipping round...")
            return
            
        # generate context vectors for each candidate index
        if verbose: print(f"Generating context vectors...")
        self.context_vectors = self.generate_context_vectors(self.candidate_indexes, False)

        # select best configuration/super-arm based on C^2 LinUCB
        if verbose: print(f"Selecting best configuration...")
        selected_indexes = self.select_best_configuration(self.context_vectors, self.candidate_indexes, verbose)

        end_time = time.perf_counter()
        rec_time = (end_time - start_time) * 1000

        # materialize the selected indexes
        if verbose: print(f"Materializing selected indexes...")
        mat_time = self.materialize_indexes(selected_indexes, verbose)
        self.materialization_time.append(mat_time)

        # execute the mini workload and observe bandit arm rewards
        if verbose: print(f"Executing mini workload...")
        index_reward, exec_time = self.execute_mini_workload(mini_workload, self.candidate_indexes, selected_indexes, restart_server, verbose)
        self.execution_time.append(exec_time)

        start_time = time.perf_counter()

        # update the LinUCB model parameters
        if verbose: print(f"Updating parameters...")
        self.update_parameters(self.candidate_indexes, index_reward, self.context_vectors, verbose)

        end_time = time.perf_counter()
        rec_time += (end_time - start_time) * 1000
        self.recommendation_time.append(rec_time)

        # update selected indexes for next round
        self.selected_indexes_last_round = selected_indexes

        print(f"\nRound {self.current_round} completed! Recommendation time: {rec_time:.2f} ms, Materialization time: {mat_time:.2f} ms, Execution time: {exec_time:.2f} ms")



    # identify new query templates from the mini workload and update stats
    def identify_new_query_templates(self, mini_workload, verbose):
        for query in mini_workload:
            if query.template_id not in self.query_store:
                # add to query store
                self.query_store[query.template_id] = query
                self.query_store[query.template_id].frequency = 1
                self.query_store[query.template_id].first_seen = self.current_round
            else:
                # update stats    
                self.query_store[query.template_id].frequency += 1
                self.query_store[query.template_id].query_string = query.query_string   # keep most recent query string
            
            self.query_store[query.template_id].last_seen = self.current_round

        if verbose: 
            print(f"\tQuery Store:")
            for query in self.query_store.values():
                print(f"\t\tTemplate ID: {query.template_id}, Frequency: {query.frequency}, First Seen: {query.first_seen}, Last Seen: {query.last_seen}")

    # select queries of interest (QoIs) from past workload and use them to extract candidate indexes, i.e. bandit arms
    def select_queries_of_interest(self, mini_workload, verbose):
        # select queries of interest (QoIs) from past workload and use them to extract candidate indexes, i.e. bandit arms
        QoIs = []
        for query in self.query_store.values():
            print(f"Query template: {query.template_id}, Query last seen: {query.last_seen}")
            # select queries that have been seen in the last qoi_memory rounds, excluding new templates from current round
            if self.current_round - query.last_seen <= self.qoi_memory and query.first_seen != self.current_round:
                QoIs.append(query)

        if verbose:
            print(f"\tQueries of Interest (QoIs):")
            for query in QoIs:
                print(f"\t\tTemplate ID: {query.template_id}, Frequency: {query.frequency}, First Seen: {query.first_seen}, Last Seen: {query.last_seen}")        

        return QoIs        


    # extract candidate indexes from QoIs
    def extract_candidate_indexes(self, QoIs, verbose):
        # extract candidate indexes from QoIs
        candidate_indexes = {}
        for query_object in QoIs:
            # extract indexes from the query
            indexes = extract_query_indexes(query_object,  self.MAX_INDEX_COLUMNS, self.INCLUDE_COLS)
            for index in indexes:
                if index not in candidate_indexes:
                    candidate_indexes[index.index_id] = index

        # get hypothetical sizes of all new candidate indexes not in the cache
        new_indexes = [index for index in candidate_indexes.values() if index.index_id not in self.index_size]
        if new_indexes:
            conn = create_connection()
            new_index_sizes = get_hypothetical_index_sizes(conn, new_indexes)
            close_connection(conn)    
            for index_id, size in new_index_sizes.items():
                self.index_size[index_id] = size

        if verbose:
            print(f"\tCandidate Indexes:")
            for index in candidate_indexes.values():
                print(f"\t\tIndex ID: {index.index_id}, Index Columns: {index.index_columns}, Include Columns: {index.include_columns}, Size: {self.index_size[index.index_id]} Mb")            

        return candidate_indexes   


    # generate context vectors for each candidate index
    def generate_context_vectors(self, candidate_indexes, verbose):
        # generate column context
        column_context_vectors = self.generate_column_context(candidate_indexes)
        # generate derived context
        derived_context_vectors = self.generate_derived_context(candidate_indexes)
        # concatenate column and derived context vectors
        context_vectors = np.hstack((derived_context_vectors, column_context_vectors))

        if verbose:
            print(f"\tContext Vectors:")
            for index, context_vector in zip(candidate_indexes.values(), context_vectors):
                print(f"\t\tIndex ID: {index.index_id}, Context Vector: {context_vector}")

        return context_vectors     


    # generate column context vectors
    def generate_column_context(self, candidate_indexes):
        column_context_vectors = []
        for index in candidate_indexes.values():
            if index.index_id in self.column_context_cache:
                # check if column context is already cached
                column_context = self.column_context_cache[index.index_id]
            else:
                # create separate encoding segments for index columns and include columns    
                index_column_context_vector = np.zeros(len(self.columns_to_idx), dtype=float)
                include_column_context_vector = np.zeros(len(self.columns_to_idx), dtype=float)

                for position, column in enumerate(index.index_columns):
                    # encode index columns with exponentially decreasing weight based on position (since order matters)
                    index_column_context_vector[self.columns_to_idx[column]] = 1 / (10**position)

                for position, column in enumerate(index.include_columns):
                    # encode include columns with uniform weights (since order doesn't matter)
                    include_column_context_vector[self.columns_to_idx[column]] = 1

                # concatenate index columns and include columns
                column_context = np.hstack((index_column_context_vector, include_column_context_vector))    

                # cache the column context
                self.column_context_cache[index.index_id] = column_context

            column_context_vectors.append(column_context)
    
        column_context_vectors = np.vstack(column_context_vectors)

        return column_context_vectors


    # generate derived context vectors
    def generate_derived_context(self, candidate_indexes):
        derived_context_vectors = []
        for index in candidate_indexes.values():
            derived_context = np.zeros(2, dtype=float)
            derived_context[0] = self.index_average_reward[index.index_id]

            if index.index_id not in self.selected_indexes_last_round:
                derived_context[1] = self.index_size[index.index_id] / self.database_size
            else:
                # if candidate index was selected in the last round, then set to 0   
                derived_context[1] = 0
            derived_context_vectors.append(derived_context)

        derived_context_vectors = np.vstack(derived_context_vectors)

        return derived_context_vectors


    # select best configuration/super-arm using C^2 LinUCB
    def select_best_configuration(self, context_vectors, candidate_indexes, verbose):
        # compute linUCB parameters
        V_inv = np.linalg.inv(self.V)
        theta = V_inv @ self.b
        # rescale the weight corresponding to second component of the derived context vector, i.e. creation cost
        # (this is useful for reducing the impact of creation cost on the selection process)
        theta[1] = theta[1]/self.creation_time_reduction_factor
        # compute expected rewards
        expected_rewards = (context_vectors @ theta).reshape(-1)
        # estimate upper confidence bound/variance
        variances = self.alpha * np.sqrt(np.diag(context_vectors @ V_inv @ context_vectors.T))
        # compute upper bounds
        upper_bounds = expected_rewards + variances
        # convert to dict
        upper_bounds = {index_id: upper_bound for index_id, upper_bound in zip(candidate_indexes.keys(), upper_bounds)}
        # solve 0-1 knapsack problem to select best configuration
        selected_indexes = self.solve_knapsack(upper_bounds, candidate_indexes)
        # convert to dict
        selected_indexes = {index.index_id: index for index in selected_indexes}
        # decay alpha
        self.alpha = self.alpha * self.alpha_decay_rate

        for index_id in selected_indexes:
            self.index_selected_rounds[index_id].append(self.current_round)

        if verbose:
            # sort indexes by decreasing order of upper bound
            sorted_indexes = sorted(upper_bounds, key=upper_bounds.get, reverse=True)
            print(f"\tUpper Bounds:")
            for index_id in sorted_indexes:
                print(f"\t\tIndex ID: {index_id}, Upper Bound: {upper_bounds[index_id]}")

            print(f"\tSelected Indexes:")
            print(f"\t\t{list(selected_indexes.keys())}")    

        return selected_indexes


    # greedy 1/2 approximation algorithm for knapsack problem
    def solve_knapsack(self, upper_bounds, candidate_indexes):
        # keep at most self.MAX_INDEXES_PER_TABLE indexes per table
        table_index_counts = defaultdict(int)    

        # compute estimated reward upper bound to index size ratio
        ratios = {index_id: upper_bound / self.index_size[index_id] for index_id, upper_bound in upper_bounds.items()}
        # sort indexes by decreasing order of ratio
        sorted_indexes = sorted(ratios, key=ratios.get, reverse=True)
        # select indexes greedily to fit within memory budget
        selected_indexes = []
        memory_used = 0
        for index_id in sorted_indexes:
            if memory_used + self.index_size[index_id] <= self.config_memory_MB:
                # keep at most self.MAX_INDEXES_PER_TABLE indexes per table
                if table_index_counts[candidate_indexes[index_id].table_name] < self.MAX_INDEXES_PER_TABLE:
                    selected_indexes.append(candidate_indexes[index_id])
                    memory_used += self.index_size[index_id]
                    table_index_counts[candidate_indexes[index_id].table_name] += 1
            
            if memory_used >= self.config_memory_MB:
                break 

        return selected_indexes


    # materialize the selected indexes
    def materialize_indexes(self, selected_indexes, verbose):
        indexes_added = [index for index in selected_indexes.values() if index.index_id not in self.selected_indexes_last_round]
        indexes_dropped = [index for index in self.selected_indexes_last_round.values() if index.index_id not in selected_indexes]
        if verbose:
            print(f"\tIndexes Added: {[index.index_id for index in indexes_added]}")
            print(f"\tIndexes Dropped: {[index.index_id for index in indexes_dropped]}")

        # drop indexes that are no longer selected
        conn = create_connection()
        bulk_drop_indexes(conn, indexes_dropped)
        close_connection(conn)

        # materialize the selected indexes
        conn = create_connection()
        bulk_create_indexes(conn, indexes_added)
        close_connection(conn)

        # update the cached index sizes with actual sizes
        self.current_round_index_creation_time = {} # reset this
        materialization_time = 0
        for index in indexes_added:
            self.index_size[index.index_id] = index.size
            self.index_creation_time[index.index_id].append((self.current_round, index.creation_time))
            self.current_round_index_creation_time[index.index_id] = index.creation_time
            materialization_time += index.creation_time
   
        return materialization_time    


    # execute the mini workload and observe bandit arm rewards
    def execute_mini_workload(self, mini_workload, candidate_indexes, selected_indexes, restart_server, verbose):
        total_execution_time = 0
        execution_info = []
        for i,query in enumerate(mini_workload):
            if restart_server:
                # restart the server before each query execution
                restart_postgresql()
            if verbose: print(f"Executing query# {i+1}", end='\r', flush=True)
            # execute the query and observe the reward
            conn = create_connection()
            execution_time, results_rows, table_access_info, index_access_info, bitmap_heapscan_info = execute_query(conn, query.query_string, with_explain=True,  return_access_info=True, print_results=False)
            close_connection(conn)
            execution_info.append((execution_time, table_access_info, index_access_info))
            total_execution_time += execution_time

        if verbose:
            print(f"\tExecution Info:")
            for i, (query, info) in enumerate(zip(mini_workload, execution_info)):
                print(f"\t\tQuery# {i+1} , Execution Time: {info[0]}, Table Access Info: {info[1]}, Index Access Info: {info[2]}, Bitmap Heap Scan Info: {bitmap_heapscan_info}")

        # extract index scan times and table sequential scan times from the execution info and shape index reward
        index_reward = {}
        for (execution_time, table_access_info, index_access_info) in execution_info:
            for table_name, table_info in table_access_info.items():
                self.table_scan_times[table_name].append(table_info["actual_total_time"])

            for index_id, index_info in index_access_info.items():
                scan_type = index_info["scan_type"]
                bitmap_heap_scan = False
                if scan_type == 'Bitmap Index Scan':
                    # for bitmap index scans, the query plan does not provide table name so we use the table name from the actual index object 
                    table_name = candidate_indexes[index_id].table_name
                    # bitmap index scan is typically followed by a bitmap heap scan on the same table, so need to get the scan time from that heap scan
                    bitmap_heap_scan_time = bitmap_heapscan_info[table_name]["actual_total_time"] if table_name in bitmap_heapscan_info else 0
                    index_scan_time = index_info["actual_total_time"] + bitmap_heap_scan_time
                    bitmap_heap_scan = True                    
                else:
                    table_name = index_info["table"]
                    index_scan_time = index_info["actual_total_time"]
                    
                # save to index usage stats
                self.index_usage_rounds[index_id].append((self.current_round, index_scan_time))                
                # craft the reward signal for this index:
                # reward = gain - creation_cost
                # where gain = sequential scan time without index - access time with index
                # and creation_cost = index creation time if index was created in this round otherwise 0
                max_table_scan_time = max(self.table_scan_times[table_name][-self.TABLE_SCAN_TIME_HISTORY:]) 
                gain = max_table_scan_time - index_scan_time # using max observed table scan time for a more optimistic gain estimate
                creation_time = self.current_round_index_creation_time.get(index_id, 0)
                if verbose: 
                    if bitmap_heap_scan:
                        print(f"\t\t\tAccumulating reward for index: {index_id}, table: {table_name}, scan type:{scan_type}, index scan time: {index_scan_time}, bitmap heap scan time: {bitmap_heap_scan_time}, gain: {gain}, creation time: {creation_time}")
                    else:
                        print(f"\t\t\tAccumulating reward for index: {index_id}, table: {table_name}, scan type:{scan_type}, index scan time: {index_scan_time}, gain: {gain}, creation time: {creation_time}")
                
                # save gain and creation cost as separate components, accumulated gain over multiple uses
                if index_id in index_reward:
                    index_reward[index_id] = (index_reward[index_id][0] + gain, index_reward[index_id][1])
                else:
                    index_reward[index_id] = (gain, -creation_time)    


        # for indexes that were selected on this round but not used, set gain to 0
        unused_indexes = set(selected_indexes.keys()) - set(index_reward.keys())
        for index_id in unused_indexes:
            index_reward[index_id] = (0, -self.current_round_index_creation_time.get(index_id,0))

        if verbose:
            print("\tIndex Rewards:")
            for index_id, reward in index_reward.items():
                print(f"\t\tIndex ID: {index_id}, Gain: {reward[0]}, Creation Time: {reward[1]}")

        return index_reward, total_execution_time


    # update the LinUCB model parameters
    def update_parameters(self, candidate_indexes, index_reward, context_vectors, verbose):
        for i, (index_id, index) in enumerate(candidate_indexes.items()):
            if index.index_id in index_reward:
                reward = index_reward[index.index_id]
                # update moving average of index rewards
                self.index_average_reward[index.index_id] = (self.index_average_reward[index.index_id] + reward[0])/2    
                context_vector = context_vectors[i]

                # update V and b for creation cost reward component
                if reward[1] != 0:
                    creation_reward_context = np.zeros_like(context_vector)
                    creation_reward_context[1] = context_vector[1]  
                    self.V += np.outer(creation_reward_context, creation_reward_context)   
                    self.b += reward[1] * creation_reward_context.reshape(-1, 1)

                # update V and b for gain reward component
                if reward[0] != 0:
                    context_vector[1] = 0
                    gain_reward_context = context_vector
                    self.V += np.outer(gain_reward_context, gain_reward_context)  
                    self.b += reward[0] * gain_reward_context.reshape(-1, 1)


    # get total recommendation,  materialization and query execution times
    def get_total_times(self):
        return sum(self.recommendation_time), sum(self.materialization_time), sum(self.execution_time)



""" 
    No Index Baseline
"""

def execute_workload_noIndex(workload, drop_indexes=False, restart_server=False):
    if drop_indexes:
        print(f"*** Dropping all existing indexes...")
        # drop all existing indexes
        conn = create_connection()
        drop_all_indexes(conn)
        close_connection(conn)

    print(f"Executing workload without any indexes...")
    # execute workload without any indexes
    total_time = 0
    for i, query_object in enumerate(workload):
        if restart_server:
            # restart the server before each query execution
            restart_postgresql()
        
        print(f"\nExecuting query# {i+1}...")
        conn = create_connection()
        execution_time, rows, table_access_info, index_access_info, bitmap_heapscan_info = execute_query(conn, query_object.query_string, with_explain=True, return_access_info=True)
        close_connection(conn)
        execution_time /= 1000
        print(f"\tExecution_time: {execution_time} s, Indexes accessed: {list(index_access_info.keys())}\n")
        total_time += execution_time

    print(f"Total execution time for workload without any indexes: {total_time} seconds")
    
    return total_time    
