"""
    CC-MAB Algorithm
"""


import numpy as np
import os
import sys
from tqdm import tqdm
import itertools
from collections import defaultdict
import datetime

script_directory = os.path.dirname(os.path.abspath(__file__))
target_subdirectory_path = os.path.join(script_directory, 'database')
sys.path.append(target_subdirectory_path)
from utils import *



"""
   Index class definition
"""
class Index:
    def __init__(self, table_name, index_id, index_columns, size, include_columns=(), value=None, payload_only=False):
        self.table_name = table_name
        self.index_id = index_id
        self.index_columns = index_columns
        self.size = size
        self.include_columns = include_columns
        self.value = value
        self.query_template_ids = None
        self.clustered_index_time = None
        self.context_vector_columns = None
        self.payload_only = payload_only
        self.average_observed_reward = 0 

    def __str__(self):
        return f"Index({self.table_name}, {self.index_id}, {self.index_columns}, {self.include_columns}, {self.size}, {self.value})"



class MAB:
    def __init__(self, alpha=1.0, vlambda=0.5, alpha_decay_rate=1.0):
        # define Lin UCB parameters
        self.alpha = alpha     # UCB exploration parameter
        self.vlambda = vlambda # regularization parameter
        self.alpha_decay_rate = alpha_decay_rate  # decay rate for alpha

        # get all columns and drop all non clustered indices
        connection = start_connection()
        self.all_columns, self.num_columns = get_all_columns(connection)
        remove_all_nonclustered_indexes(connection)
        close_connection(connection)

        self.context_size = self.num_columns + self.num_columns + 2  # index_columns + include_columns + derived_context

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

        # constants
        self.MAX_INDEX_COLUMNS = 6
        self.MAX_INCLUDE_COLUMNS=4 
        self.SMALL_TABLE_IGNORE=1000
        self.TABLE_MIN_SELECTIVITY=0.5
        self.MAX_INDEXES_PER_TABLE=3


    # perform one round of the MAB algorithm
    def step_round(self, new_miniworkload, current_time_step, query_memory=1, config_memory_budget_MB=128, verbose=False):
        print(f"Running MAB agent for round {current_time_step}...")    

        # open a connection to the database
        connection = start_connection()

        start_time = datetime.datetime.now()
        # identify newly observed query templates and update statistics for previously observed templates
        print(f"Identifying new query templates and updating statistics...")
        queries_current_round = []
        num_new = 0
        for query in new_miniworkload:
            query_template_id = query.template_id   
            if query_template_id in self.query_store:
                # update statistics for previously seen query template
                self.query_store[query_template_id].frequency += 1
                self.query_store[query_template_id].last_seen = current_time_step
                self.query_store[query_template_id].query_string = query.query_string # keep most recenet query string for this template
                if self.query_store[query_template_id].first_seen == -1:
                    self.query_store[query_template_id].first_seen = current_time_step
            else:
                # create a new query template object for newly observed query template
                q = Query(connection, query.template_id, query.query_string,  query.payload, query.predicates, query.order_by, query.group_by, time_created=current_time_step)      
                # generate context vector for the query
                q.context_vector = self.generate_context_vector_query(q)
                # add the query template to the query store
                self.query_store[query_template_id] = q  
                num_new += 1
            queries_current_round.append(self.query_store[query_template_id])

        print(f"Number of new query templates added: {num_new}")

        # select query templates of interest (QoI) for this round, these will be the query templates that have been seen recently
        # (i.e. within the last query_memory rounds in the past) but not in the current round 
        if verbose: print(f"Selecting queries of interest...")
        QoI_list = []
        for query_template_id, query_obj in self.query_store.items():
            # if the query is seen recently, i.e. within the last query_memory rounds in the past, then add it to the QoI list
            if (current_time_step - query_obj.last_seen <= query_memory) and (0 <= query_obj.first_seen < current_time_step):
                QoI_list.append(query_obj)

            # otherwise mark it as not of interest, with last_seen set to -1    
            elif current_time_step - query_obj.last_seen > query_memory:
                query_obj.last_seen = -1

        print(f"Number of queries of interest: {len(QoI_list)}")        

        end_time = datetime.datetime.now()
        recommendation_time = end_time - start_time

        if len(QoI_list) > 0:
            start_time = datetime.datetime.now()
            # generate candidate indices using predicate and payload information from QoI
            print(f"Generating candidate indices...")
            self.candidate_index_arms = self.generate_candidate_indices(connection, QoI_list, verbose=verbose)
            # generate context vectors
            print(f"Generating context vectors...")    
            context_vectors = self.generate_contexts(connection, self.candidate_index_arms, selected_indices_past=self.selected_indices_last_round)
            
            # select best configuration
            print(f"Selecting best configuration...")
            selected_indices = self.select_best_configuration(context_vectors, self.candidate_index_arms, config_memory_budget_MB, verbose=verbose)

            end_time = datetime.datetime.now()
            recommendation_time += (end_time - start_time)    

            # materialze configuration then execute current round queries and get observed rewards
            print(f"Materializing configuration and executing queries...")
            total_execution_cost, creation_cost, index_rewards = self.materialize_execute(connection, selected_indices, queries_current_round, self.candidate_index_arms, verbose=verbose)

            # update parameters
            print(f"Updating MAB parameters...")
            self.update_parameters(selected_indices, index_rewards, self.candidate_index_arms)
            self.selected_indices_last_round = selected_indices

        else:
            selected_indices = {}
            self.candidate_index_arms = {}                          

            # materialze configuration then execute current round queries and get observed rewards
            print(f"Materializing configuration and executing queries...")
            total_execution_cost, creation_cost, index_rewards = self.materialize_execute(connection, selected_indices, queries_current_round, self.candidate_index_arms, verbose=verbose)

        # sort the index select counts from highest to lowest
        sorted_index_selection_count = {k: v for k, v in sorted(self.index_selection_count.items(), key=lambda item: item[1], reverse=True)}
        print(f"Index selection counts: {sorted_index_selection_count}")

        # close the connection
        close_connection(connection)

        print(f"\nRound completed. Time taken for recommendation: {recommendation_time} seconds.\n")

    # materialize a new configuration, executes new batch of queries and returns the observed rewards
    def materialize_execute(self, connection, selected_indices, queries_current_round, candidate_index_arms, table_scan_time_length=1000, verbose=False):
       
        if len(selected_indices) > 0:
            # find which new indices to add and which ones to remove
            existing_indices = set(self.selected_indices_last_round.keys())
            new_indices = set(selected_indices.keys())
            indices_to_remove = existing_indices - new_indices
            indexes_to_remove = [self.selected_indices_last_round[index_id] for index_id in list(indices_to_remove)]
            indexes_to_add = new_indices - existing_indices
            indexes_to_add = [selected_indices[index_id] for index_id in list(indexes_to_add)]
            
            # materialize the configuration
            creation_cost = bulk_create_drop_nonclustered_indexes(connection, indexes_to_add, indexes_to_remove, verbose=verbose)
        else:
            indexes_to_add = []
            indexes_to_remove = []
            creation_cost = {}

        # execute the queries
        total_execution_cost = 0
        index_rewards = {}
        for i, query in enumerate(queries_current_round):
            execution_cost, non_clustered_index_usage, clustered_index_usage = execute_query(query.query_string, connection) #, verbose=verbose)
            total_execution_cost += execution_cost
            # aggregate index usage data
            non_clustered_index_usage = self.merge_index_use(non_clustered_index_usage)
            clustered_index_usage = self.merge_index_use(clustered_index_usage)
            
            current_clustered_index_scans = {}
            if clustered_index_usage:
                # get clustered index scan times for each table
                for index_scan in clustered_index_usage:
                    table_name = index_scan[0]
                    current_clustered_index_scans[table_name] = index_scan[1]
                    if table_name not in query.table_scan_times:
                        query.table_scan_times[table_name] = []
                    if table_name not in self.table_scan_times:
                        self.table_scan_times[table_name] = []
                    if len(query.table_scan_times[table_name]) < table_scan_time_length:
                        # record the scan time for this table if the table scan time history is not full
                        query.table_scan_times[table_name].append(index_scan[1])
                        self.table_scan_times[table_name].append(index_scan[1])

            if non_clustered_index_usage:
                #if verbose:
                #    print(f"Processing non-clustered index usage for query with template id {query.template_id}")
                #    print(f"Non-clustered index usage: {non_clustered_index_usage}")
                
                # get the access counts for each table
                index_access_counts = defaultdict(int)
                for index_use in non_clustered_index_usage:
                    index_name = index_use[0]
                    table_name = candidate_index_arms[index_name].table_name
                    index_access_counts[table_name] += 1

                #if verbose:
                #    print(f"Index access counts: {index_access_counts}")    
                # get the index scan times for each table
                #if verbose:
                #    print(f"Gathering index scan times...")
                for index_use in non_clustered_index_usage:
                    #if verbose: print(f"Processing index usage: {index_use}")
                    index_name = index_use[0]
                    table_name = candidate_index_arms[index_name].table_name
                    if len(query.index_scan_times[table_name]) < table_scan_time_length:
                        # record the index scan time for this table if the table scan time history is not full
                        if table_name not in query.index_scan_times:
                            query.index_scan_times[table_name] = [index_use[1]]
                        else:
                            query.index_scan_times[table_name].append(index_use[1])
                    if table_name not in query.table_scan_times:
                        query.table_scan_times[table_name] = []
                    table_scan_time = query.table_scan_times[table_name]
                    if len(table_scan_time) > 0:
                        # compute the reward for this index as the difference between (max) full-table-scan time and index scan time
                        # averaged over the number of times the index was accessed
                        temp_reward = max(table_scan_time) - index_use[1]
                        temp_reward /= index_access_counts[table_name]
                    elif len(self.table_scan_times[table_name]) > 0:
                        temp_reward = max(self.table_scan_times[table_name]) - index_use[1]
                        temp_reward /= index_access_counts[table_name]
                    else:
                        raise ValueError("Index scan time could not be determined for a query.")    

                    # add clustered index scan time as negative reward
                    if table_name in current_clustered_index_scans:
                        temp_reward -= current_clustered_index_scans[table_name]/index_access_counts[table_name]

                    # the reward is a tuple containing a component for index-scan time during query execution and another for index creation
                    if index_name not in index_rewards:
                        index_rewards[index_name] = [temp_reward, 0]
                    else:
                        index_rewards[index_name][0] += temp_reward    

            #print(f"Non clustered index usage for query# {i+1}: {non_clustered_index_usage}")    

        # add in the index creation cost to the reward
        for index_id in creation_cost:
            if index_id not in index_rewards:
                index_rewards[index_id] = [0, -creation_cost[index_id]]
            else:
                index_rewards[index_id][1] -= creation_cost[index_id]

        print(f"Num indexes added:{len(indexes_to_add)}, Num indexes removed: {len(indexes_to_remove)} Index creation costs: {creation_cost}")
        print(f"Observed Rewards for indexes: {index_rewards}\n")
        print(f"Total execution cost: {total_execution_cost}, Total index creation cost: {sum(creation_cost.values())}\n")
 
        return total_execution_cost, creation_cost, index_rewards


    # incrementally aggregate index usage data from multiple query executions      
    def merge_index_use(self, index_uses):
        d = defaultdict(list)
        for index_use in index_uses:
            if index_use[0] not in d:
                d[index_use[0]] = [0] * (len(index_use) - 1)
            d[index_use[0]] = [sum(x) for x in zip(d[index_use[0]], index_use[1:])]
        return [tuple([x]+y) for x, y in d.items()]


    """ 
        Candidate index generation
    """
    
    # Given a query, generate candidate indices based on the predicates and payload columns in the query.
    def generate_candidate_indices_from_predicates(self, connection, query, verbose=False):
        # get all tables in the db
        tables = get_all_tables(connection)
        if verbose:
            print(f"Tables:")
            for key in tables:
                print(tables[key])
            print(f"Query predicates: {query.predicates}")
            print(f"Query payload: {query.payload}")    

        query_template_id = query.template_id
        query_predicates = query.predicates
        query_payload = query.payload
        
        indices = {}

        # indexes on predicate columns only
        for table_name, table_predicates in query_predicates.items():
            table = tables[table_name]
            if verbose: print(f"\nTable --> {table_name}, Predicate Columns --> {set(table_predicates)}, table row count --> {table.row_count}")
            
            # identify include columns
            include_columns = []
            if table_name in query_payload:
                include_columns = list(set(query_payload[table_name]) - set(table_predicates))
            
            if verbose: 
                print(f"Include columns: {include_columns}")
                print(f"Query selectivity: {query.selectivity[table_name]}")


            # check if conditions for cheap full table scan are met
            if table.row_count < self.SMALL_TABLE_IGNORE or ((query.selectivity[table_name] > self.TABLE_MIN_SELECTIVITY) and (len(include_columns)>0)):
                if verbose: print(f"Full table scan for table: {table_name} is cheap, skipping")
                continue

            # generate all possible permutations of predicate columns, from single column up to MAX_COLUMNS-column indices
            table_predicates = list(table_predicates.keys())  #[0:6]
            col_permutations = []
            for num_columns in range(1, min(self.MAX_COLUMNS+1, len(table_predicates)+1)):
                col_permutations = col_permutations + list(itertools.permutations(table_predicates, num_columns)) 
            
            if verbose: print(f"Column permutations: \n{col_permutations}")

            # assign an id and value to each index/column permutation
            for cp in col_permutations:
                index_id = get_index_id(cp, table_name)
                
                if index_id not in indices:
                    index_size = get_estimated_index_size(connection, table_name, cp)
                    if verbose:  print(f"index_id: {index_id}, index columns: {cp}, index size: {index_size:.2f} Mb")
                    # assign value...

                    # create index object
                    indices[index_id] = Index(table_name, index_id, cp, index_size)

        # indexes on columns that are in the payload but not in the predicates
        for table_name, table_payload in query_payload.items():
            table = tables[table_name]
            if verbose: print(f"\nTable --> {table_name}, Payload Columns --> {set(table_predicates)}, table row count --> {table.row_count}")
            
            # skip if any of the payload columns for this table are in the predicates
            if table_name in query_predicates:
                if verbose: print(f"Payload columns are in the predicates, skipping")
                continue

            # check if conditions for cheap full table scan are met
            if table.row_count < self.SMALL_TABLE_IGNORE:
                if verbose: print(f"Full table scan for table: {table_name} is cheap, skipping")
                continue   

            # don't need to consider permutations here, just create an index with all payload columns in given order
            index_id = get_index_id(table_payload, table_name)
            if index_id not in indices:
                index_size = get_estimated_index_size(connection, table_name, table_payload)
                print(f"index_id: {index_id}, index columns: {table_payload}, index size: {index_size:.2f} Mb")
                # assign value... (will assign less value to these indices as they are less useful compared to predicate indices)
                indices[index_id] = Index(table_name, index_id, table_payload, index_size, payload_only=True)

        # indexes with include columns
        for table_name, table_predicates in query_predicates.items():
            table = tables[table_name]
            if verbose: print(f"\nTable --> {table_name}, Predicate Columns --> {set(table_predicates)}, table row count --> {table.row_count}")
            
            # check if conditions for cheap full table scan are met
            if table.row_count < self.SMALL_TABLE_IGNORE:
                if verbose: print(f"Full table scan for table: {table_name} is cheap, skipping")
                continue  

            # identify include columns
            include_columns = []
            if table_name in query_payload:
                include_columns = sorted(list(set(query_payload[table_name]) - set(table_predicates)))

            if len(include_columns)>0:    
                if verbose: print(f"Include columns: {include_columns}")
                # generate all possible permutations of predicate columns
                table_predicates = list(table_predicates.keys())  
                col_permutations = list(itertools.permutations(table_predicates, len(table_predicates))) 
                #col_permutations = list(itertools.permutations(table_predicates, MAX_COLUMNS+1)) 
                if verbose: print(f"Column permutations: \n{col_permutations}")

                # assign an id and value to each index/column permutation
                for cp in col_permutations:
                    index_id = get_index_id(cp, table_name, include_columns)
                    if index_id not in indices:
                        index_size = get_estimated_index_size(connection, table_name, list(cp) + include_columns)
                        if verbose: print(f"index_id: {index_id}, index columns: {cp}, include columns: {include_columns}, index size: {index_size:.2f} Mb")
                        # assign value...
                        # create index object
                        indices[index_id] = Index(table_name, index_id, cp, index_size, tuple(include_columns))

        return indices        


    # version 2: takees into account, group by and order by columns
    def generate_candidate_indices_from_predicates_v2(self, connection, query, verbose=False):
        # get all tables in the db
        tables = get_all_tables(connection)
        if verbose:
            print(f"Tables:")
            for key in tables:
                print(tables[key])
            print(f"Query predicates: {query.predicates}")
            print(f"Query payload: {query.payload}")    

        query_template_id = query.template_id
        query_predicates = query.predicates
        query_payload = query.payload
        query_group_by = query.group_by
        query_order_by = query.order_by
        
        indices = {}

        # indexes on predicate, group by and order by columns - without and with include columns
        for table_name, table_predicates in query_predicates.items():
            table = tables[table_name]
            if verbose: print(f"\nTable --> {table_name}, Predicate Columns --> {set(table_predicates)}, table row count --> {table.row_count}")
            
            # identify include columns
            include_columns = []
            if table_name in query_payload:
                include_columns = list(set(query_payload[table_name]) - set(table_predicates))
            
            if verbose: 
                print(f"Include columns: {include_columns}")
                print(f"Query selectivity: {query.selectivity[table_name]}")


            # check if conditions for cheap full table scan are met
            #if table.row_count < SMALL_TABLE_IGNORE or ((query.selectivity[table_name] > TABLE_MIN_SELECTIVITY) and (len(include_columns)>0)):
            if table.row_count < self.SMALL_TABLE_IGNORE:
                if verbose: print(f"Full table scan for table: {table_name} is cheap, skipping")
                continue

            # generate all possible permutations of predicate, group_by and order_by columns, from single column up to MAX_COLUMNS-column indices
            predicates_columns = list(table_predicates.keys()) 
            group_by_columns = query_group_by[table_name] if table_name in query_group_by else []
            order_by_columns = query_order_by[table_name] if table_name in query_order_by else []
            order_by_columns = [col[0] for col in order_by_columns]

            if verbose:
                print(f"Predicates columns: {predicates_columns}")
                print(f"Group by columns: {group_by_columns}")
                print(f"Order by columns: {order_by_columns}")

            combined_columns = list(set(predicates_columns + group_by_columns + order_by_columns))

            col_permutations = []
            for num_columns in range(1, min(self.MAX_INDEX_COLUMNS+1, len(combined_columns)+1)):
                col_permutations = col_permutations + list(itertools.permutations(combined_columns, num_columns)) 
            
            if verbose: print(f"Column permutations: \n{col_permutations}")

            # assign an id and value to each index/column permutation
            for cp in col_permutations:
                if verbose: print(f"Considering index columns permutation: {cp}")
                index_id_wo_include = get_index_id(cp, table_name)
                
                if index_id_wo_include not in indices:
                    index_size = get_estimated_index_size(connection, table_name, cp)
                    if verbose:  print(f"index_id: {index_id_wo_include}, index columns: {cp}, index size: {index_size:.2f} Mb")
                    # assign value...
                    # create index object
                    indices[index_id_wo_include] = Index(table_name, index_id_wo_include, cp, index_size)


                # remove any include columns already in the index columns 
                include_columns_filtered = list(set(include_columns) - set(cp))
                if len(include_columns_filtered)>0:  

                    if verbose: print(f"Include columns: {include_columns_filtered}")
                    # generate all combinations of include columns from 1 to MAX_COLUMNS, ignore ordering
                    include_col_combinations = []
                    for num_columns in range(1, min(self.MAX_INCLUDE_COLUMNS+1, len(include_columns_filtered)+1)):
                        include_col_combinations = include_col_combinations + list(itertools.combinations(include_columns_filtered, num_columns))
                    
                    if verbose: print(f"Include column combinations: \n{include_col_combinations}")

                    for cc in include_col_combinations:
                        if verbose: print(f"Considering include columns combination: {cc}")
                        index_id = get_index_id(cp, table_name, cc)
                        if index_id not in indices:
                            index_size = get_estimated_index_size(connection, table_name, list(cp) + list(cc))
                            if verbose: print(f"index_id: {index_id}, index columns: {cp}, include columns: {cc}, index size: {index_size:.2f} Mb")
                            # assign value...
                            # create index object
                            indices[index_id] = Index(table_name, index_id, cp, index_size, cc)


        # indexes on columns that are in the payload but not in the predicates
        for table_name, table_payload in query_payload.items():
            table = tables[table_name]
            if verbose: print(f"\nTable --> {table_name}, Payload Columns --> {set(table_predicates)}, table row count --> {table.row_count}")
            
            # skip if any of the payload columns for this table are in the predicates
            if table_name in query_predicates:
                if verbose: print(f"Payload columns are in the predicates, skipping")
                continue

            # check if conditions for cheap full table scan are met
            if table.row_count < self.SMALL_TABLE_IGNORE:
                if verbose: print(f"Full table scan for table: {table_name} is cheap, skipping")
                continue   

            # don't need to consider permutations here, just create an index with all payload columns in given order
            index_id = get_index_id(table_payload, table_name)
            if index_id not in indices:
                index_size = get_estimated_index_size(connection, table_name, table_payload)
                print(f"index_id: {index_id}, index columns: {table_payload}, index size: {index_size:.2f} Mb")
                # assign value... (will assign less value to these indices as they are less useful compared to predicate indices)
                indices[index_id] = Index(table_name, index_id, table_payload, index_size, payload_only=True)

        return indices        



    # Given a miniworkload, which is a list of query objects, generate candidate indices
    def generate_candidate_indices(self, connection, queries, verbose=False):
        print(f"Generating candidate indices for {len(queries)} queries...")
        candidate_index_arms = {} 
        for query in tqdm(queries, desc="Processing queries"):
            query_candidate_indices = self.generate_candidate_indices_from_predicates_v2(connection, query, verbose=verbose)
            for index_id, index in query_candidate_indices.items():
                if index_id not in candidate_index_arms:
                    # initialization
                    index.query_template_ids = set()
                    index.clustered_index_time = 0
                    candidate_index_arms[index_id] = index

                # add the maximum table scan time for the table associated with this index and query template
                if index.table_name in query.table_scan_times:
                    candidate_index_arms[index_id].clustered_index_time += max(query.table_scan_times[index.table_name])  
                candidate_index_arms[index_id].query_template_ids.add(query.template_id)

        print(f"Generated {len(candidate_index_arms)} candidate indices")
          
        return candidate_index_arms


    """
        Context vector generation
    """

    # generate query context vector
    def generate_context_vector_query(self, query_object):
        # check if cached context vector is available
        if query_object.context_vector:
            return query_object.context_vector
        
        # for each column in the query predicates, set the corresponding index in the context vector to 1 
        context_vector = np.zeros(shape=(self.context_size,1), dtype=float)
        for table_name, columns in query_object.predicates.items():
            for column in columns:
                context_vector[self.columns_to_idx[column]] = 1

        # cache the context vector
        query_object.context_vector = context_vector

        return context_vector


    # generate columns piece
    def generate_context_vector_columns_index(self, index, columns_to_idx):
        # return the cached context vector if available
        if index.context_vector_columns:
            return index.context_vector_columns

        index_column_context_vector = np.zeros(len(columns_to_idx), dtype=float)
        include_column_context_vector = np.zeros(len(columns_to_idx), dtype=float)
        
        # context vector for index columns
        for j, column_name in enumerate(index.index_columns):
            column_position_in_index = j
            index_column_context_vector[columns_to_idx[column_name]] = 1/(10**column_position_in_index)
        # context vector for include columns
        for j, column_name in enumerate(index.include_columns):
            include_column_context_vector[columns_to_idx[column_name]] = 1

        # concatenate the two context vectors
        context_vector = np.hstack((index_column_context_vector, include_column_context_vector))
        # cache the context vector
        index.context_vector_columns = context_vector    

        return context_vector    


    def generate_context_vector_columns(self, index_arms, columns_to_idx):
        # stack up the context vectors for all indices into a single matrix
        context_vectors = np.vstack([self.generate_context_vector_columns_index(index, columns_to_idx) for index in index_arms.values()])
        
        return context_vectors


    # generate derived piece
    def generate_context_vector_derived(self, connection, index_arms, selected_indices_past):
        database_size = get_database_size(connection)
        derived_context_vectors = np.zeros((len(index_arms), 2), dtype=float)
        
        for i, index in enumerate(index_arms.values()):    
            # the first derived context component will be the average observed reward for this index
            derived_context_vectors[i,0] =  index.average_observed_reward
            
            # the second derived context component will be used to estimate index creation cost as a linear function of index usage
            # i.e. estimated index creation cost is the index size multiplied by a corresponding weight/parameter
            if index.index_id not in selected_indices_past:
                # if the index was not selected in the last round then it will be created, otherwise assigned a value of 0    
                derived_context_vectors[i,1] =  index.size/database_size
        
        return derived_context_vectors


    # generate context vectors (selected_indices_past should be a list of index ids)
    def generate_contexts(self, connection, index_arms, selected_indices_past=[]):
        columns_context_vectors = self.generate_context_vector_columns(index_arms, self.columns_to_idx)
        derived_context_vectors = self.generate_context_vector_derived(connection, index_arms, selected_indices_past)

        # concatenate the derived and column context vectors
        return np.hstack((derived_context_vectors, columns_context_vectors))
    

    """ 
        Selection of best configuration/super-arm. Here, we use LinUCB algorithm to select the best configuration: 
            * first estimate the upper bound for the estimated expected reward for each index
            * then (approximately) solve 0-1 knapsack problem to select a subset of indices which
                maximizes estimated total expected reward while satisfying constraint on config memory budget 

    """
    def select_best_configuration(self, context_vectors, index_arms, config_memory_budget_MB, creation_cost_reduction_factor=10, verbose=False):
        self.context_vectors = context_vectors
        # compute parameters vector
        V_inv = np.linalg.inv(self.V)        
        theta = V_inv @ self.b 

        print(f"Theta - parameter vector: \n{theta.reshape(-1)}")
        
        # rescale the parameter corresponding to the size of the index
        theta[1] = theta[1]/creation_cost_reduction_factor  
        # estimate the expected reward upper bound for each arm/index
        expected_reward = (context_vectors @ theta).reshape(-1)
        # estimate upper confidence bound
        confidence_bounds = self.alpha * np.sqrt(np.diag(context_vectors @ V_inv @ context_vectors.T))
        self.upper_bounds = expected_reward + confidence_bounds   
        #if verbose:
            #print(f"expected rewards shape {expected_reward.shape}")
            #print(f"confidence bounds shape {confidence_bounds.shape}")
            #print(f"Expected reward upper bounds: {self.upper_bounds}")
        
        #################################
        # find most negative upper bound
        min_upper_bound = min(self.upper_bounds)
        # make all upper bounds non-negative
        if min_upper_bound < 0:
            self.upper_bounds = self.upper_bounds - min_upper_bound 
        
        # if verbose: print(f"Upper bounds after shifting: {self.upper_bounds}")
        #################################

        # filter out indices that exceed the memory budget
        filtered_upper_bounds = {}
        for i, index_id in enumerate(index_arms.keys()):
            if index_arms[index_id].size <= config_memory_budget_MB:
                filtered_upper_bounds[index_id] = self.upper_bounds[i]

        if verbose: print(f"Filtered upper bounds: {filtered_upper_bounds}")

        self.upper_bounds = filtered_upper_bounds        

        # keep only top MAX_INDEXES_PER_TABLE indices for each table in the filtered upper bounds
        table_indexes = defaultdict(list)
        for index_id in self.upper_bounds:
            table_indexes[index_arms[index_id].table_name].append(index_id)

        # sort the indices for each table in descending order of upper bound and keep only the top MAX_INDEXES_PER_TABLE indices for each table
        table_filtered_upper_bounds = {}
        for table_name, indexes in table_indexes.items():
            table_indexes[table_name] = sorted(indexes, key=self.upper_bounds.get, reverse=True)[:self.MAX_INDEXES_PER_TABLE]
            table_filtered_upper_bounds.update({index_id: self.upper_bounds[index_id] for index_id in table_indexes[table_name]})

        self.upper_bounds = table_filtered_upper_bounds    
        print(f"Table filtered indexes: {table_indexes}")
        print(f"Number of candidate indexes after filtration: {len(self.upper_bounds)}")

        # solve knapsack problem to select the best configuration
        selected_indices = self.knapsack_solver(index_arms, config_memory_budget_MB, verbose)
        #selected_indices = self.subset_subtotal_solver(index_arms, config_memory_budget_MB, verbose)
        
        selected_indices = {index.index_id: index for index in selected_indices}

        # update the index selection count
        for index_id in selected_indices:
            self.index_selection_count[index_id] += 1

        # decay alpha
        self.alpha = self.alpha * self.alpha_decay_rate    

        print(f"Best configuration contains {len(selected_indices)} indices: \n{list(selected_indices.keys())}")    
        self.best_configuration = selected_indices

        return selected_indices


    # greedy 1/2 approximation algorithm for 0-1 knapsack problem     
    def knapsack_solver(self, index_arms, config_memory_budget_MB, verbose=False):
        # compute the ratio of the upper bound to the size of the index
        ratios = {index_id: upper_bound/index_arms[index_id].size for index_id, upper_bound in self.upper_bounds.items()}
        # sort the indices in descending order of ratio
        sorted_indices = sorted(ratios, key=ratios.get, reverse=True)
        # select the indices that fit within the memory budget
        selected_indices = []
        total_size = 0
        for index_id in sorted_indices:
            if total_size + index_arms[index_id].size <= config_memory_budget_MB:
                selected_indices.append(index_arms[index_id])
                total_size += index_arms[index_id].size
            else:
                break    

        return selected_indices        

    # sorts the indices based on the upper bound and selects the top-k indices that fit within the memory budget
    def subset_subtotal_solver(self, index_arms, config_memory_budget_MB, verbose=False):
        sorted_indices = sorted(self.upper_bounds, key=self.upper_bounds.get, reverse=True)
        selected_indices = []
        total_size = 0
        for index_id in sorted_indices:
            if total_size + index_arms[index_id].size <= config_memory_budget_MB:
                selected_indices.append(index_arms[index_id])
                total_size += index_arms[index_id].size
            else:
                break

        return selected_indices        


    # updates matrix V and vector b based on the observed rewards
    def update_parameters(self, selected_indices, observed_rewards, candidate_index_arms):
        index_name_to_idx = {index_id: i for i, index_id in enumerate(candidate_index_arms.keys())}
        for index_id in selected_indices.keys():
            # get the context vector for this index
            context_vector = self.context_vectors[index_name_to_idx[index_id]]
            # get the observed reward for this index
            if index_id in observed_rewards:
                index_reward = observed_rewards[index_id]
            else:
                # no reward observed for this index
                index_reward = [0, 0]

            # update the moving average observed reward for this index over all rounds
            candidate_index_arms[index_id].average_observed_reward = (index_reward[0] + candidate_index_arms[index_id].average_observed_reward)/2

            # separate out the update for the index creation cost/reward component
            temp_context = np.zeros_like(context_vector)
            temp_context[1] = context_vector[1]
            context_vector[1] = 0

            # update the matrix V and vector b for non-index creation reward component
            self.V += np.outer(context_vector, context_vector)
            self.b += index_reward[0] * context_vector.reshape(-1,1)

            # update the matrix V and vector b for index creation cost/reward component
            # this is done separately to avoid the index creation cost/reward component from dominating the update
            self.V += np.outer(temp_context, temp_context)
            self.b += index_reward[1] * temp_context.reshape(-1,1)             

        # reset the context vectors and upper bounds
        self.context_vectors = None
        self.upper_bounds  = None


















