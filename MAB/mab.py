"""
    CC-MAB Algorithm
"""


import numpy as np
import os
import sys
from tqdm import tqdm
import itertools
from collections import defaultdict

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
    def __init__(self, alpha=1.0, vlambda=0.5):
        # define Lin UCB parameters
        self.alpha = alpha     # UCB exploration parameter
        self.vlambda = vlambda # regularization parameter

        # get all columns
        connection = start_connection()
        self.all_columns, self.num_columns = get_all_columns(connection)
        close_connection(connection)

        self.context_size = self.num_columns + 2  # columns + derived
        self.columns_to_idx = {}
        i = 0
        for table_name, columns in self.all_columns.items():
            for column in columns:
                self.columns_to_idx[column] = i
                i += 1

        self.idx_to_columns = {v: k for k, v in self.columns_to_idx.items()}   

        # initialize matrix V and vector b
        self.V = np.eye(self.context_size) * self.vlambda
        self.b = np.zeros(shape=(self.context_size, 1))
        self.context_vectors = None
        self.upper_bounds  = None
        self.index_selection_count = defaultdict(int)
        

    """ 
        Candidate index generation
    """

    
    # Given a query, generate candidate indices based on the predicates and payload columns in the query.
    def generate_candidate_indices_from_predicates(self, connection, query, MAX_COLUMNS=6, SMALL_TABLE_IGNORE=10000, TABLE_MIN_SELECTIVITY=0.2, verbose=False):
        # get all tables in the db
        tables = get_all_tables(connection)
        if verbose:
            print(f"Tables:")
            for key in tables:
                print(tables[key])

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
            if table.row_count < SMALL_TABLE_IGNORE or ((query.selectivity[table_name] > TABLE_MIN_SELECTIVITY) and (len(include_columns)>0)):
                if verbose: print(f"Full table scan for table: {table_name} is cheap, skipping")
                continue

            # generate all possible permutations of predicate columns, from single column up to MAX_COLUMNS-column indices
            table_predicates = list(table_predicates.keys())  #[0:6]
            col_permutations = []
            for num_columns in range(1, min(MAX_COLUMNS, len(table_predicates)+1)):
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
            if table.row_count < SMALL_TABLE_IGNORE:
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
            if table.row_count < SMALL_TABLE_IGNORE:
                if verbose: print(f"Full table scan for table: {table_name} is cheap, skipping")
                continue  

            # identify include columns
            include_columns = []
            if table_name in query_payload:
                include_columns = sorted(list(set(query_payload[table_name]) - set(table_predicates)))

            if len(include_columns)>0:    
                if verbose: print(f"Include columns: {include_columns}")

                # generate all possible permutations of predicate columns
                table_predicates = list(table_predicates.keys())#[0:6]
                #col_permutations = list(itertools.permutations(table_predicates, len(table_predicates))) 
                col_permutations = list(itertools.permutations(table_predicates, MAX_COLUMNS)) 
                
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



    # Given a miniworkload, which is a list of query objects, generate candidate indices
    def generate_candidate_indices(self, connection, miniworkload, verbose=False):
        print(f"Gnereting candidate indices for {len(miniworkload)} queries...")
        index_arms = {} 
        for query in tqdm(miniworkload, desc="Processing queries"):
            query_candidate_indices = self.generate_candidate_indices_from_predicates(connection, query, verbose=verbose)
            for index_id, index in query_candidate_indices.items():
                if index_id not in index_arms:
                    # initialization
                    index.query_template_ids = set()
                    index.clustered_index_time = 0
                    index_arms[index_id] = index

                # add the maximum table scan time for the table associated with this index and query template
                index_arms[index_id].clustered_index_time += max(query.table_scan_times[index.table_name] if query.table_scan_times[index.table_name] else 0)   
                index_arms[index_id].query_template_ids.add(query.template_id)

        return index_arms



    """
        Context vector generation
    """

    # generate columns piece
    def generate_context_vector_columns_index(self, index, columns_to_idx):
        # return the cached context vector if available
        if index.context_vector_columns:
            return index.context_vector_columns

        context_vector = np.zeros(len(columns_to_idx), dtype=float)
        for j, column in enumerate(index.index_columns):
            context_vector[columns_to_idx[column]] = 10**(-j)

        # cache the context vector
        index.encode_context_vector = context_vector    

        return context_vector    


    def generate_context_vector_columns(self, index_arms, columns_to_idx):
        # stack up the context vectors for all indices into a single matrix
        context_vectors = np.vstack([self.generate_context_vector_columns_index(index, columns_to_idx) for index in index_arms.values()])
        
        return context_vectors


    # generate derived piece
    def generate_context_vector_derived(self, connection, index_arms, selected_indices_last_round):
        database_size = get_database_size(connection)
        derived_context_vectors = np.zeros((len(index_arms), 2), dtype=float)
        
        for i, index in enumerate(index_arms.values()):    
            # the first derived context component will be the average observed reward for this index
            derived_context_vectors[i,0] =  index.average_observed_reward
            
            # the second derived context component will be used to estimate index creation cost as a linear function of index usage
            # i.e. estimated index creation cost is the index size multiplied by a corresponding weight/parameter
            if index.index_id not in selected_indices_last_round:
                # if the index was not selected in the last round then it will be created, otherwise assigned a value of 0    
                derived_context_vectors[i,1] =  index.size/database_size
        
        return derived_context_vectors


    # generate context vectors (selected_indices_last_round should be a list of index ids)
    def generate_contexts(self, connection, index_arms, selected_indices_last_round=[]):
        columns_context_vectors = self.generate_context_vector_columns(index_arms, self.columns_to_idx)
        derived_context_vectors = self.generate_context_vector_derived(connection, index_arms, selected_indices_last_round)

        # concatenate the derived and column context vectors
        return np.hstack((derived_context_vectors, columns_context_vectors))
    

    """ 
        Selection of best configuration/super-arm
            Here, we use LinUCB algorithm to select the best configuration: 
                * first estimate the upper bound for the estimated expected reward for each index
                * then (approximately) solve 0-1 knapsack problem to select a subset of indices which
                  maximizes estimated total expected reward while satisfying constraint on config memory budget 

    """
    def select_best_configuration(self, context_vectors, index_arms, max_config_size_MB=1024, creation_cost_reduction_factor=3, verbose=False):
        self.context_vectors = context_vectors
        V_inv = np.linalg.inv(self.V)
        
        # compute parameters vector
        theta = V_inv @ self.b 
        # rescale the parameter corresponding to the size of the index
        theta[1] = theta[1]/creation_cost_reduction_factor  
        # estimate the expected reward upper bound for each arm/index
        expected_reward = (context_vectors @ theta).reshape(-1)
        # estimate upper confidence bound
        confidence_bounds = self.alpha * np.sqrt(np.diag(context_vectors @ V_inv @ context_vectors.T))
        # confidence_bounds = self.alpha * np.sqrt(np.einsum('ij,jk,ik->i', context_vectors, V_inv, context_vectors))
        self.upper_bounds = expected_reward + confidence_bounds   
        if verbose:
            print(f"expected rewards shape {expected_reward.shape}")
            print(f"confidence bounds shape {confidence_bounds.shape}")
            #print(f"Expected reward upper bounds: {self.upper_bounds}")

        # solve knapsack problem to select the best configuration
        selected_indices = self.knapsack_solver(index_arms, max_config_size_MB, verbose)

        # update the index selection count
        for index in selected_indices:
            self.index_selection_count[index.index_id] += 1

        return selected_indices


    # greedy 1/2 approximation algorithm for 0-1 knapsack problem     
    def knapsack_solver(self, index_arms, max_config_size_MB, verbose=False):
        # compute the ratio of the upper bound to the size of the index
        ratios = self.upper_bounds / np.array([index.size for index in index_arms.values()])
        if verbose:
            print(f"Ratios shape: {ratios.shape}")

        # sort the indices in descending order of ratio
        sorted_indices = np.argsort(ratios)[::-1] 
        # Extract the sorted index_arms values based on sorted_indices
        sorted_index_arms = [list(index_arms.values())[i] for i in sorted_indices]
        # select the indices that fit within the memory budget
        selected_indices = []
        current_memory_usage = 0
        for index in sorted_index_arms:
            index_size = index.size
            if current_memory_usage + index_size <= max_config_size_MB:
                selected_indices.append(index)
                current_memory_usage += index_size
            else:
                break    

        return selected_indices        


    # updates matrix V and vector b based on the observed rewards
    def update_parameters(self, selected_indices, observed_rewards, candidate_index_arms):
        index_name_to_idx = {index.index_id: i for i, index in enumerate(candidate_index_arms)}
        for index in selected_indices:
            # get the context vector for this index
            context_vector = self.context_vectors[index_name_to_idx[index.index_id]]
            # get the observed reward for this index
            if index.index_id in observed_rewards:
                index_reward = observed_rewards[index.index_id]
            else:
                index_reward = (0, 0)

            # update the moving average observed reward for this index over all rounds
            candidate_index_arms[index.index_id].average_observed_reward = (index_reward[0] + candidate_index_arms[index.index_id].average_observed_reward)/2

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













