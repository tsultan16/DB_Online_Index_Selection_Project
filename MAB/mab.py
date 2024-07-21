"""
    CC-MAB Algorithm
"""


import numpy as np
import os
import sys
from tqdm import tqdm
import itertools

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
        self.index_usage_last = 0 

    def __str__(self):
        return f"Index({self.table_name}, {self.index_id}, {self.index_columns}, {self.include_columns}, {self.size}, {self.value})"



class MAB:
    def __init__(self):
        pass


    """
        Given a query, generate candidate indices based on the predicates and payload columns in the query.
    """
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



    """
        Given a miniworkload, which is a list of query objects, generate candidate indices
    """
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
    def generate_context_vector_derived(self, connection, index_arms):
        database_size = get_database_size(connection)
        derived_context_vectors = np.zeros((len(index_arms), 2), dtype=float)
        for i, index in enumerate(index_arms.values()):
            derived_context_vectors[i,0] =  index.index_usage_last
            derived_context_vectors[i,1] =  index.size/database_size
        
        return derived_context_vectors
