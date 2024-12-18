"""
    MAB Implementation (v2)
"""


import sys
import os
import operator

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

        # drop all existing indexes
        print("Dropping all existing secondary indexes...")
        conn = create_connection()
        drop_all_indexes(conn)
        close_connection(conn)

        # get database size
        if MAB.database_size_cache is None:
            conn = create_connection()
            MAB.database_size_cache = get_database_size(conn)
            close_connection(conn)
        else:
            print("Found database size in cache")    
        self.database_size = MAB.database_size_cache    
        print(f"Database size: {self.database_size} MB")
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
        self.index_query_templates = defaultdict(set) # tracks which query templates an index was extracted from
        self.table_scan_times = {}
        for table_name in self.tables.keys():
            self.table_scan_times[table_name] = self.tables[table_name]["sequential_scan_time"]
        # keep copy of indexes selected in previous round and current round candidates
        self.selected_indexes_last_round = {}
        self.candidate_indexes = {}
        self.indexes_currently_materialized = {}
 
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
        if verbose: print(f"\nIdentifying new query templates from the mini workload...")
        self.identify_new_query_templates(mini_workload, verbose)

        # select queries of interest (QoIs) from past workload and use them to extract candidate indexes, i.e. bandit arms
        if verbose: print(f"\nSelecting QoIs and extracting candidate indexes...")
        QoIs = self.select_queries_of_interest(mini_workload, verbose)
        self.candidate_indexes = self.extract_candidate_indexes(QoIs, verbose)

        # generate context vectors for each candidate index
        if verbose: print(f"\nGenerating context vectors...")
        self.context_vectors = self.generate_context_vectors(self.candidate_indexes, False)

        # select best configuration/super-arm based on C^2 LinUCB
        if verbose: print(f"\nSelecting best configuration...")
        selected_indexes = self.select_best_configuration(self.context_vectors, self.candidate_indexes, verbose)

        end_time = time.perf_counter()
        rec_time = (end_time - start_time) * 1000

        # materialize the selected indexes
        if verbose: print(f"\nMaterializing selected indexes...")
        mat_time = self.materialize_indexes(selected_indexes, verbose)
        self.materialization_time.append(mat_time)

        # execute the mini workload and observe bandit arm rewards
        if verbose: print(f"\nExecuting mini workload...")
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

        if verbose:
            print(f"\nRound {self.current_round} completed! Recommendation time: {rec_time:.2f} ms, Materialization time: {mat_time:.2f} ms, Execution time: {exec_time:.2f} ms")
            print(f"\nIndex usage stats:")
            for index_id, usage_rounds in self.index_usage_rounds.items():
                print(f"\tIndex ID: {index_id}, Usage Count: {len(usage_rounds)}")
            print(f"\nMax table scan times:")
            for table_name, scan_times in self.table_scan_times.items():
                print(f"\tTable: {table_name}, Max Scan Time: {max(scan_times)} ms")    
            print(f"\nIndexes currently materialized: {list(self.indexes_currently_materialized.keys())}")


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
            print(f"\n\tQueries of Interest (QoIs):")
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
                # update index query template stats
                if query_object.template_id not in self.index_query_templates[index.index_id]:
                    self.index_query_templates[index.index_id].add(query_object.template_id)    

        # get hypothetical sizes of all new candidate indexes not in the cache
        new_indexes = [index for index in candidate_indexes.values() if index.index_id not in self.index_size]
        if new_indexes:
            conn = create_connection()
            new_index_sizes = get_hypothetical_index_sizes(conn, new_indexes)
            close_connection(conn)    
            for index_id, size in new_index_sizes.items():
                self.index_size[index_id] = size
                if size == 0.0:
                    raise ValueError(f"Index size is 0 for index ID: {index_id}") 

        if verbose:
            print(f"\tExtracted {len(candidate_indexes)} candidate indexes from QoIs.")
            #for index in candidate_indexes.values():
            #    print(f"\t\tIndex ID: {index.index_id}, Index Columns: {index.index_columns}, Include Columns: {index.include_columns}, Size: {self.index_size[index.#index_id]} Mb")            

        return candidate_indexes   


    # generate context vectors for each candidate index
    def generate_context_vectors(self, candidate_indexes, verbose):
        if len(candidate_indexes) == 0:
            return None
        
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
        if len(candidate_indexes) == 0:
            return {}

        # compute linUCB parameters
        V_inv = np.linalg.inv(self.V) # shape = (context_size, context_size)
        theta = V_inv @ self.b # shape = (context_size, 1)

        if verbose:
            print(f"\nTheta: {theta.reshape(-1)}")

        upper_bounds = {}
        #estimated_creation_costs = {}
        #context_components = {}
        if self.current_round == 2:    
            # assign uniform upper bounds for first round
            shuffled_indexes = list(candidate_indexes.keys())
            np.random.shuffle(shuffled_indexes)
            upper_bounds = {index_id:1 for index_id in shuffled_indexes}

        else:      
            for i, index_id in enumerate(candidate_indexes.keys()):
                context_vector = context_vectors[i].reshape(-1, 1) # shape = (context_size, 1)    
                # estimated creation cost
                creation_cost = np.ndarray.item(theta[1] * context_vector[1])
                # estimate reward excluding creation cost
                average_reward = np.ndarray.item(theta.T @ context_vector) - creation_cost
                # compute ucb
                ucb = average_reward + self.alpha * np.sqrt(np.ndarray.item(context_vector.T @ V_inv @ context_vector))
                # add in scaled creation cost
                ucb += creation_cost / self.creation_time_reduction_factor
                upper_bounds[index_id] = ucb
                #estimated_creation_costs[index_id] = creation_cost
                #context_components[index_id] = context_vector[0:2].reshape(-1)
                #print(f"\nIndex ID: {index_id}, Context: {context_vector[0:2]}, Estimated Creation Cost: {creation_cost}, Estimated Reward: {average_reward}, UCB: {ucb}")

        # solve 0-1 knapsack problem to select best configuration
        selected_indexes = self.solve_knapsack(upper_bounds, candidate_indexes, verbose)
        #selected_indexes = self.submodular_oracle(upper_bounds, candidate_indexes, verbose)
        # convert to dict
        selected_indexes = {index.index_id: index for index in selected_indexes}
        # decay alpha
        self.alpha = self.alpha * self.alpha_decay_rate

        for index_id in selected_indexes:
            self.index_selected_rounds[index_id].append(self.current_round)

        if verbose:
            # sort indexes by decreasing order of upper bound
            sorted_indexes = sorted(upper_bounds, key=upper_bounds.get, reverse=True)
            print(f"\nUpper Bounds:")
            i = 0
            for index_id in sorted_indexes:
                #print(f"\tIndex ID: {index_id}, Context:{context_components[index_id]}, Upper Bound: {upper_bounds[index_id]}, Estimated Creation Cost: {estimated_creation_costs[index_id]}")
                print(f"\tIndex ID: {index_id}, Upper Bound: {upper_bounds[index_id]}")
                i += 1
                if i == 50:
                    break

            sorted_selected_indexes = {k: v for k, v in sorted(selected_indexes.items(), key=lambda item: upper_bounds[item[0]], reverse=True)}
            # create dict mapping selected index id to upper bound, in sorted order of decreasing upper bound  
            sorted_indexes = {index_id: upper_bounds[index_id] for index_id in sorted_selected_indexes}
            print(f"\n\tSelected Indexes:")
            print(f"\t\t{list(sorted_indexes.items())}")    

        return selected_indexes



    # submodular maximization oracle for obtaining knapsack problem approximate solution
    def submodular_oracle(self, upper_bounds, candidate_indexes, verbose, low_reward_threshold=0):
        used_memory = 0
        chosen_arms = []
        arm_ucb_dict = {}
        table_count = {}

        # remove arms with low rewards
        arm_ucb_dict = {index_id:ucb for index_id, ucb in upper_bounds.items() if ucb > low_reward_threshold}

        while len(arm_ucb_dict) > 0:
            # select the arm with the highest UCB if it fits within memory budget
            max_ucb_arm_id = max(arm_ucb_dict.items(), key=operator.itemgetter(1))[0]
            if self.index_size[max_ucb_arm_id] < self.config_memory_MB - used_memory:
                chosen_arms.append(candidate_indexes[max_ucb_arm_id])
                used_memory += self.index_size[max_ucb_arm_id]
                if candidate_indexes[max_ucb_arm_id].table_name in table_count:
                    table_count[candidate_indexes[max_ucb_arm_id].table_name] += 1
                else:
                    table_count[candidate_indexes[max_ucb_arm_id].table_name] = 1

                # filter out arms that are dominated by the chosen arm
                arm_ucb_dict = self.filter_arms(arm_ucb_dict, max_ucb_arm_id, candidate_indexes, table_count, used_memory)
            else:
                # remove arm from consideration if it exceeds memory budget
                arm_ucb_dict.pop(max_ucb_arm_id)

        if verbose: print(f"\nMax memory budget: {self.config_memory_MB} MB, Used memory: {used_memory} MB")

        return chosen_arms
    

    def filter_arms(self, arm_ucb_dict, chosen_id, candidate_indexes, table_count, used_memory, prefix_length=1):
        remaining_memory = self.config_memory_MB - used_memory

        # keep at most self.MAX_INDEXES_PER_TABLE indexes per table
        reduced_arm_ucb_dict = {}
        for index_id in arm_ucb_dict:
            if not (candidate_indexes[index_id].table_name == candidate_indexes[chosen_id].table_name and table_count[candidate_indexes[index_id].table_name] >= self.MAX_INDEXES_PER_TABLE):
                reduced_arm_ucb_dict[index_id] = arm_ucb_dict[index_id]
        arm_ucb_dict = reduced_arm_ucb_dict

        """
        # remove arms that are part of the same cluster to improve diversity
        reduced_arm_ucb_dict = {}
        for index_id in arm_ucb_dict:
            if not (candidate_indexes[index_id].table_name == candidate_indexes[chosen_id].table_name and candidate_indexes[chosen_id].cluster is not None
                    and candidate_indexes[index_id].cluster == candidate_indexes[chosen_id].cluster):
                reduced_arm_ucb_dict[index_id] = arm_ucb_dict[index_id]
        arm_ucb_dict = reduced_arm_ucb_dict        
                
        # remove arms whose associated query templates have been covered by already selected arms
        reduced_arm_ucb_dict = {}
        for index_id in arm_ucb_dict:
            # get the query templates associated with the chosen arm
            query_template_ids = self.index_query_templates[chosen_id]
            for query_template_id in query_template_ids:
                if (candidate_indexes[index_id].table_name == candidate_indexes[chosen_id].table_name and 
                    candidate_indexes[chosen_id].is_include and query_template_id in self.index_query_templates[index_id]):
                    # if query template is covered by the chosen arm, then remove that template from the other arm
                    self.index_query_templates[index_id].remove(query_template_id)     

            # if there are still remaining query templates covered by this arm, then keep it
            if self.index_query_templates[index_id] != set():
                reduced_arm_ucb_dict[index_id] = arm_ucb_dict[index_id]
        arm_ucb_dict = reduced_arm_ucb_dict        
        """

        # remove arms that exceed the remaining memory budget
        arm_ucb_dict = {index_id: index for index_id, index in arm_ucb_dict.items() if self.index_size[index_id] <= remaining_memory}    

        # remove arms with matching prefix up to a prefix_length
        if len(candidate_indexes[chosen_id].index_columns) >= prefix_length:
            recuced_arm_ucb_dict = {}
            for index_id in arm_ucb_dict:
                if (candidate_indexes[index_id].table_name == candidate_indexes[chosen_id].table_name and len(candidate_indexes[index_id].index_columns) > prefix_length):
                    if candidate_indexes[index_id].index_columns[:prefix_length] != candidate_indexes[chosen_id].index_columns[:prefix_length]:
                        recuced_arm_ucb_dict[index_id] = arm_ucb_dict[index_id]
                else:
                    recuced_arm_ucb_dict[index_id] = arm_ucb_dict[index_id]        
            arm_ucb_dict = recuced_arm_ucb_dict    
               
        return arm_ucb_dict

        
    # greedy 1/2 approximation algorithm for knapsack problem
    def solve_knapsack(self, upper_bounds, candidate_indexes, verbose, remove_negative_ucb=False, ignore_size=True):
        # keep at most self.MAX_INDEXES_PER_TABLE indexes per table
        table_index_counts = defaultdict(int)    

        # compute estimated reward upper bound to index size ratio
        if remove_negative_ucb:
            if ignore_size:
                ratios = {index_id: upper_bound for index_id, upper_bound in upper_bounds.items() if upper_bound >= 0}
            else:
                ratios = {index_id: upper_bound / self.index_size[index_id] for index_id, upper_bound in upper_bounds.items() if upper_bound >= 0}
        else:
            if ignore_size:
                ratios = {index_id: upper_bound for index_id, upper_bound in upper_bounds.items()}
            else:
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

        if verbose:
            print(f"\nMax memory budget: {self.config_memory_MB} MB, Used memory: {memory_used} MB")
           
        return selected_indexes


    # materialize the selected indexes
    def materialize_indexes(self, selected_indexes, verbose):

        indexes_added = [index for index in selected_indexes.values() if index.index_id not in self.selected_indexes_last_round]
        indexes_dropped = [index for index in self.selected_indexes_last_round.values() if index.index_id not in selected_indexes]

        for index in indexes_added:
            self.indexes_currently_materialized[index.index_id] = index
        for index in indexes_dropped:
            self.indexes_currently_materialized.pop(index.index_id)

        if verbose:
            print(f"\n\tIndexes Added: {[index.index_id for index in indexes_added]}")
            print(f"\n\tIndexes Dropped: {[index.index_id for index in indexes_dropped]}\n")

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
            if verbose: print(f"\nExecuting query# {i+1}")
            # execute the query and observe the reward
            conn = create_connection()
            execution_time, results_rows, table_access_info, index_access_info, bitmap_heapscan_info = execute_query(conn, query.query_string, with_explain=True,  return_access_info=True, print_results=False)
            close_connection(conn)
            execution_info.append((execution_time, table_access_info, index_access_info, bitmap_heapscan_info))
            total_execution_time += execution_time



        if verbose:
            print(f"\nExecution Info:")
            for i, (query, info) in enumerate(zip(mini_workload, execution_info)):
                #print(f"\n\t\tQuery# {i+1} , Execution Time: {info[0]}")
                #print(f"\t\tTable Access Info: {info[1]}")
                #print(f"\t\tIndex Access Info: {info[2]}")
                #print(f"\t\tBitmap Heap Scan Info: {info[3]}")
                print(f"\nQuery# {i+1} , Execution Time: {info[0]}, Tables Accessed: {list(info[1].keys())}, Indexes Accessed: {list(info[2].keys())}, Bitmap Heap Scans : {list(bitmap_heapscan_info.keys())}")
            print()    

        # extract index scan times and table sequential scan times from the execution info and shape index reward
        index_reward = {}
        for (execution_time, table_access_info, index_access_info, bitmap_heapscan_info) in execution_info:
            for table_name, table_info in table_access_info.items():
                self.table_scan_times[table_name].append(table_info["actual_total_time"])

            if len(candidate_indexes) == 0:
                continue

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
                        print(f"Accumulating reward: {index_id}, table: {table_name}, scan type:{scan_type}, index scan time: {index_scan_time} ms, bitmap heap scan time: {bitmap_heap_scan_time} ms, gain: {gain}, creation time: {creation_time} ms\n")
                    else:
                        print(f"Accumulating reward: {index_id}, table: {table_name}, scan type:{scan_type}, index scan time: {index_scan_time} ms, gain: {gain}, creation time: {creation_time} ms\n")
                
                # save gain and creation cost as separate components, accumulated gain over multiple uses
                if index_id in index_reward:
                    index_reward[index_id] = (index_reward[index_id][0] + gain, index_reward[index_id][1])
                else:
                    index_reward[index_id] = (gain, -creation_time)    

        if len(candidate_indexes) == 0:
            return None, total_execution_time

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
        if len(candidate_indexes) == 0:
            return

        for i, index_id in enumerate(candidate_indexes.keys()):
            if index_id in index_reward:
                reward = index_reward[index_id]
                # update moving average of index rewards
                self.index_average_reward[index_id] = (self.index_average_reward[index_id] + reward[0])/2    
                context_vector = context_vectors[i]

                # update V and b for creation cost reward component
                creation_reward_context = np.zeros_like(context_vector)
                creation_reward_context[1] = context_vector[1]  
                self.V += np.outer(creation_reward_context, creation_reward_context)   
                self.b += reward[1] * creation_reward_context.reshape(-1, 1)

                # update V and b for gain reward component
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
