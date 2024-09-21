""" 
    Simple Cost Model Implementation 

    This algorithms assumes that query execution cost is dominated by disk IO. It predicts the cheapest access path for each table in the query and also estimates the total number of pages accesed in each path.    

"""

import sys
import os
import re
import numpy as np


# Add Postgres directory to sys.path
module_path = os.path.abspath('/home/tanzid/Code/DBMS/PostgreSQL')
if module_path not in sys.path:
    sys.path.append(module_path)

from pg_utils import *
from ssb_qgen_class import *



def get_page_size():
    return 8192  # 8 KB

def get_table_stats(table_name):

    conn = create_connection()

    cur = conn.cursor()

    try:
        # Execute the query to get the estimated number of rows in the table
        cur.execute(f"""
                    SELECT reltuples::bigint AS estimated_rows
                    FROM pg_class
                    WHERE relname = '{table_name}';
                    """)
        estimated_rows = cur.fetchone()[0]
    except:
        print(f"Error: Could not get the estimated number of rows in the '{table_name}' table.")
        estimated_rows = None

    try:
        # Query to get column statistics
        cur.execute(f"SELECT * FROM pg_stats WHERE tablename = '{table_name}';")
        column_stats = cur.fetchall()

        # Define the column names based on the pg_stats view
        column_names = [
                        "schemaname", "tablename", "attname", "inherited", "null_frac",
                        "avg_width", "n_distinct", "most_common_vals", "most_common_freqs",
                        "histogram_bounds", "correlation", "most_common_elems",
                        "most_common_elem_freqs", "elem_count_histogram"
                    ]

        # Organize the results into a dictionary
        stats_dict = {}
        for row in column_stats:
            column_name = row[2]  # 'attname' is the third column in the result
            stats_dict[column_name] = {column_names[i]: row[i] for i in range(len(column_names))}
    except:
        print(f"Error: Could not get the statistics for the '{table_name}' table")
        stats_dict = None

    # Close the cursor and connection
    cur.close()
    close_connection(conn)

    return stats_dict, estimated_rows



class SimpleCost:

    def __init__(self, stats, estimated_rows, sequential_scan_cost_multiplier=1.0, index_scan_cost_multiplier=2.0):
        self.sequentail_scan_cost_multiplier = sequential_scan_cost_multiplier
        self.index_scan_cost_multiplier = index_scan_cost_multiplier 
        """    
        # Get the statistics for all tables in the SSB database
        self.table_names = ["customer", "dwdate", "lineorder", "part", "supplier"]
        self.stats = {}
        self.estimated_rows = {}
        for table_name in self.table_names:
            self.stats[table_name], self.estimated_rows[table_name] = get_table_stats(table_name)
        """
        self.stats, self.estimated_rows = stats, estimated_rows

        ssb_tables, pk_columns = get_ssb_schema()
        # create a dictionary and specify whether each attribute in each table is numeric or char
        self.data_type_dict = {}
        for table_name in ["customer", "dwdate", "lineorder", "part", "supplier"]:
            for column_name, column_type in ssb_tables[table_name]:
                if ("INT" in column_type) or ("DECIMAL" in column_type) or ("BIT" in column_type):
                    self.data_type_dict[column_name] = "numeric"
                else:
                    self.data_type_dict[column_name] = "char"


    def estimate_selectivity_one_sided_range(self, attribute, boundary_value, operator, stats_dict, total_rows):
        data_type = self.data_type_dict[attribute]
        # Get the column statistics
        stats = stats_dict[attribute]
        histogram_bounds = stats['histogram_bounds']
        n_distinct = stats['n_distinct']
        most_common_vals = stats['most_common_vals']
        most_common_freqs = stats['most_common_freqs']

        # Convert most_common_values string to list of correct data type
        if most_common_vals:
            if data_type == 'numeric':
                most_common_vals = [float(x) for x in most_common_vals.strip('{}').split(',')]
            elif data_type == 'char':
                most_common_vals = [x for x in most_common_vals.strip('{}').split(',')]
            else:
                raise ValueError("Data type not supported, needs to be either numeric or char")

        # Convert negative n_distinct to an absolute count
        if n_distinct < 0:
            n_distinct = -n_distinct * total_rows

        selectivity = 0.0

        # Check for overlap with most common values
        if most_common_vals:
            for val, freq in zip(most_common_vals, most_common_freqs):
                if (operator == '>' and val > boundary_value) or (operator == '<' and val < boundary_value):
                    selectivity += freq

        if histogram_bounds is not None:
            if data_type == 'numeric':
                histogram_bounds = [float(x) for x in histogram_bounds.strip('{}').split(',')]
            elif data_type == 'char':
                histogram_bounds = [x for x in histogram_bounds.strip('{}').split(',')]
            else:
                raise ValueError("Data type not supported, needs to be either numeric or char")

            total_bins = len(histogram_bounds) - 1

            # Iterate over bins, find overlapping bins
            for i in range(total_bins):
                bin_lower_bound = histogram_bounds[i]
                bin_upper_bound = histogram_bounds[i + 1]

                if data_type == 'numeric':
                    # Check for range overlap
                    if (operator == '>' and boundary_value < bin_upper_bound) or (operator == '<' and boundary_value > bin_lower_bound):
                        # Calculate the overlap fraction within this bin
                        if operator == '>':
                            overlap_min = max(boundary_value, bin_lower_bound)
                            overlap_fraction = (bin_upper_bound - overlap_min) / (bin_upper_bound - bin_lower_bound)
                        else:  # operator == '<'
                            overlap_max = min(boundary_value, bin_upper_bound)
                            overlap_fraction = (overlap_max - bin_lower_bound) / (bin_upper_bound - bin_lower_bound)

                        # Accumulate to the total selectivity
                        selectivity += overlap_fraction * (1.0 / total_bins)

                elif data_type == 'char':
                    if (operator == '>' and boundary_value < bin_upper_bound) or (operator == '<' and boundary_value > bin_lower_bound):
                        # assume the whole bin overlaps
                        overlap_fraction = 1.0
                        # Accumulate to the total selectivity
                        selectivity += overlap_fraction * (1.0 / total_bins)



        if selectivity == 0.0:
            # If no overlap with most common values or histogram bins, assume uniform distribution and estimate selectivity
            selectivity = 1.0 / n_distinct

        return selectivity


    def estimate_selectivity_range(self, attribute, value_range, stats_dict, total_rows):
        data_type = self.data_type_dict[attribute]
        # get the column statistics
        stats = stats_dict[attribute]
        # get the histogram bounds
        histogram_bounds = stats['histogram_bounds']
        n_distinct = stats['n_distinct']
        most_common_vals = stats['most_common_vals']
        most_common_freqs = stats['most_common_freqs']


        #print(f"Histogram bounds: {histogram_bounds}")
        #print(f"Most common values: {most_common_vals}")
        #print(f"Most common frequencies: {most_common_freqs}")

        # convert most_common_values string to list of correct data type
        if most_common_vals:
            if data_type == 'numeric':
                most_common_vals = [float(x) for x in most_common_vals.strip('{}').split(',')]
            elif data_type == 'char':
                most_common_vals = [x for x in most_common_vals.strip('{}').split(',')]    
            else:
                raise ValueError("Data type not supported, needs ot be either numeric or char")

        # Convert negative n_distinct to an absolute count
        if n_distinct < 0:
            n_distinct = -n_distinct * total_rows

        min_value = value_range[0]
        max_value = value_range[1]
        selectivity = 0.0

        # check for overlap with most common values
        if most_common_vals:
            for val, freq in zip(most_common_vals, most_common_freqs):
                if min_value <= val <= max_value:
                    selectivity += freq    

        if histogram_bounds is not None:
            if data_type == 'numeric':
                histogram_bounds = [float(x) for x in histogram_bounds.strip('{}').split(',')] # convert to list of integers
            elif data_type == 'char':
                histogram_bounds = [x for x in histogram_bounds.strip('{}').split(',')]
            else:
                raise ValueError("Data type not supported, needs ot be either numeric or char")    

            total_bins = len(histogram_bounds) - 1

            # iterate over bins, find overlapping bins
            for i in range(total_bins):
                bin_lower_bound = histogram_bounds[i]
                bin_upper_bound = histogram_bounds[i+1]

                # check for range overlap
                if min_value < bin_lower_bound or max_value > bin_upper_bound:
                    # does not overlap
                    continue    

                if data_type == 'numeric':
                    # calculate the overlap fraction within this bin
                    overlap_min = max(min_value, bin_lower_bound)
                    overlap_max = min(max_value, bin_upper_bound)
                    overlap_fraction = (overlap_max - overlap_min) / (bin_upper_bound - bin_lower_bound)

                    #print(f"Overlap fraction for bin {i}: {overlap_fraction}")
                    #print(f"Bin bounds: {bin_lower_bound}, {bin_upper_bound}")

                    # accumulate to the total selectivity
                    # Assume each bin represents an equal fraction of the total rows
                    selectivity += overlap_fraction * (1.0 / total_bins)

                elif data_type == 'char':
                    # assume the whole bin overlaps
                    overlap_fraction = 1.0
                    # accumulate to the total selectivity
                    # Assume each bin represents an equal fraction of the total rows
                    selectivity += overlap_fraction * (1.0 / total_bins)


        if selectivity == 0.0:
            # if no overlap with most common values or histogram bins, assume uniform distribution and estimate selectivity
            selectivity = 1.0 / n_distinct       

        return selectivity


    def estimate_selectivity_eq(self, attribute, value, stats_dict):
        data_type = self.data_type_dict[attribute]
        # get the column statistics
        stats = stats_dict[attribute]
        # get the histogram bounds
        histogram_bounds = stats['histogram_bounds']
        n_distinct = stats['n_distinct']
        most_common_vals = stats['most_common_vals']
        most_common_freqs = stats['most_common_freqs']

        # convert most_common_values string to list of correct data type
        if most_common_vals:
            if data_type == 'numeric':
                most_common_vals = [float(x) for x in most_common_vals.strip('{}').split(',')]
            elif data_type == 'char':
                most_common_vals = [x for x in most_common_vals.strip('{}').split(',')]    
            else:
                raise ValueError("Data type not supported, needs ot be either numeric or char")

        # first check if the value is in the most common values
        if most_common_vals and value in most_common_vals:
            selectivity = most_common_freqs[most_common_vals.index(value)] 
            return selectivity

        # if not a common value, estimate using n_distinct
        if n_distinct < 0:
            n_distinct = -n_distinct

        selectivity = 1.0 / n_distinct    

        if histogram_bounds is not None:
            if data_type == 'numeric':
                histogram_bounds = [float(x) for x in histogram_bounds.strip('{}').split(',')] # convert to list of integers
            elif data_type == 'char':
                histogram_bounds = [x for x in histogram_bounds.strip('{}').split(',')]
            else:
                raise ValueError("Data type not supported, needs ot be either numeric or char")    

            total_bins = len(histogram_bounds) - 1

            # iterate over bins, find bin that contains the value
            for i in range(total_bins):
                bin_lower_bound = histogram_bounds[i]
                bin_upper_bound = histogram_bounds[i+1]

                if data_type == 'numeric':
                    # check for range overlap
                    if bin_lower_bound <= value <= bin_upper_bound:
                        bin_width = bin_upper_bound - bin_lower_bound
                        if bin_width > 0:
                            # assume uniform distribution within this bin and calculate selectivity
                            uniform_selectivity = 1.0 / (bin_width*total_bins)
                            selectivity = min(selectivity, uniform_selectivity)
                        break    

                elif data_type == 'char':
                    # check for range overlap
                    if bin_lower_bound <= value <= bin_upper_bound:
                        # assume the whole bin overlaps
                        selectivity = 1.0 / total_bins
                        break        

        return selectivity


    def estimate_selectivity_or(self, attribute, value, stats_dict):
        combined_selectivity = 0.0
        individual_selectivities = []

        # for each value in the IN list, estimate the selectivity separately
        for val in value:
            individual_selectivities.append(self.estimate_selectivity_eq(attribute, val, stats_dict))

        # compute combined selectivities using inclusion-exclusion principle and assuming independence
        for selectivity in individual_selectivities:
            combined_selectivity += selectivity 

        overlap_adjustment = 1.0
        for selectivity in individual_selectivities:
            overlap_adjustment *= (1.0 - selectivity)

        combined_selectivity -= overlap_adjustment   

        # make sure the combined selectivity is between 0 and 1
        combined_selectivity = max(0.0, min(combined_selectivity, 1.0))

        return combined_selectivity 


    def estimate_selectivity(self, attribute, operator, value, stats_dict, total_rows):
        if operator == 'eq':
            return self.estimate_selectivity_eq(attribute, value, stats_dict)
        elif operator == 'range':
            return self.estimate_selectivity_range(attribute, value, stats_dict, total_rows)
        elif operator == '<' or operator == '>':
            return self.estimate_selectivity_one_sided_range(attribute, value, operator, stats_dict, total_rows)
        elif operator == 'or':
            return self.estimate_selectivity_or(attribute, value, stats_dict)    
        else:
            raise ValueError(f"Operator '{operator}' not supported, needs to be either 'eq', 'range', or 'or'")
            

    # predict the cheapest access paths for executing the query with the given indexes    
    def predict(self, query_object, indexes, verbose=False):
        # extract tables and associated columns
        self.tables = {}
        for table_name in query_object.payload:
            self.tables[table_name] = query_object.payload[table_name]

        for table_name in query_object.predicates:
            if table_name not in self.tables:
                self.tables[table_name] = []
            self.tables[table_name] = list(set(self.tables[table_name] + query_object.predicates[table_name]))

        # extract the payload
        payload = query_object.payload
        # extract the predicates
        self.id2predicate = {}
        predicate_dict = query_object.predicate_dict
        predicates = {}
        # map predicate ids to predicate strings
        #id = 0
        for table_name in predicate_dict:
            predicates[table_name] = []
            for pred in predicate_dict[table_name]:
                id = len(self.id2predicate)
                predicates[table_name].append(id)
                self.id2predicate[id] = pred
                #id += 1

        if verbose:
            print(f"Tables and columns: {self.tables}")   
            print(f"Payload: {payload}")
            print(f"Predicates:")
            for table_name, predicate_list in predicates.items():
                print(f"\n{table_name}")
                for predicate_id in predicate_list:
                    print(f"\t{self.id2predicate[predicate_id]}")        

        # enumerate the access paths for each table
        access_paths, predicates = self.enumerate_access_paths(indexes, predicates, payload, verbose)
        # find the cheapest access paths for each table
        cheapest_table_access_path = self.find_cheapest_paths(access_paths, predicates, indexes, self.stats, self.estimated_rows, self.sequentail_scan_cost_multiplier, self.index_scan_cost_multiplier, verbose=verbose)
        
        if verbose:
            print(f"\nCheapest access paths: ")
            for table, (path, cost) in cheapest_table_access_path.items():
                print(f"Table: {table}, Cheapest path: {path}, Cost: {cost}")   
        total_cost = sum(cost for path, cost in cheapest_table_access_path.values())
        indexes_used = []
        for table, (path, cost) in cheapest_table_access_path.items():
            if path['scan_type'] == 'Index Scan' or path['scan_type'] == 'Index Only Scan':
                indexes_used.append(path['index_id'])

        return total_cost, indexes_used
    

    def enumerate_access_paths(self, indexes, predicates, payload, verbose):
        # extract join predicate columns
        join_predicates = {}
        join_predicates_temp = {}
        for table_name in predicates:
            table_preicates = predicates[table_name]
            for pred_id in table_preicates:
                pred = self.id2predicate[pred_id]
                if pred['join'] == True:
                    if table_name not in join_predicates_temp:
                        join_predicates_temp[table_name] = []
                    join_predicates_temp[table_name].append(pred['column'])
                    # add the other table's column to the join predicate list
                    other_table_column = pred['value']
                    # search for the table name containing the other column
                    for other_table_name in self.tables:
                        if other_table_column in self.tables[other_table_name]:
                            if other_table_name not in join_predicates_temp:
                                join_predicates_temp[other_table_name] = []
                            join_predicates_temp[other_table_name].append(other_table_column)
                            join_pred = pred.copy()
                            join_pred['column'] = other_table_column
                            join_pred['value'] = pred['column']
                            new_id = len(self.id2predicate)
                            self.id2predicate[new_id] = join_pred 
                            if other_table_name not in join_predicates:
                                join_predicates[other_table_name] = []
                            join_predicates[other_table_name].append(new_id)
                            break

        if verbose:
            print(f"Predicates: {predicates}")
            print(f"Join predicates: {join_predicates}")

        # add join predicates to the main predicate dictionary
        for table_name in join_predicates:
            if table_name not in predicates:
                predicates[table_name] = []
            predicates[table_name] = list(set(predicates[table_name] + join_predicates[table_name]))

        if verbose:
            print(f"Updated predicates: {predicates}")

        access_paths = {}
        for table_name in self.tables:
            if table_name in predicates:
                #table_predicate_cols = [pred['column'] for pred in predicates[table_name] if pred['join'] == False]
                table_predicate_cols = [self.id2predicate[pred_id]['column'] for pred_id in predicates[table_name]]
            if table_name in payload:
                table_payload_cols = [col for col in payload[table_name] if col in self.tables[table_name]]   
            if table_name in join_predicates_temp:
                join_predicate_cols = join_predicates_temp[table_name]
            else:
                join_predicate_cols = []    
            
            relevant_predicate_cols = set(table_predicate_cols).union(join_predicate_cols)
            table_access_paths = [{'scan_type': 'Sequential Scan'}]
            if verbose:
                print(f"\nTable predicate columns for {table_name}: {table_predicate_cols}")
                print(f"Relevant predicate columns for {table_name}: {relevant_predicate_cols}")
                print(f"Payload columns for {table_name}: {table_payload_cols}")
            for index in indexes.values():
                if index.table_name == table_name:
                    if verbose: print("Checking index: ", index.index_id)
                    # Check if index scan is possible
                    if set(index.index_columns).intersection(relevant_predicate_cols):
                        table_access_paths.append({'scan_type': 'Index Scan', 'index_id': index.index_id})
                        if verbose: print("Index scan possible!")
                    # Check if index only scan is possible
                    if set(index.index_columns).issuperset(relevant_predicate_cols) and set(
                        list(index.index_columns) + list(index.include_columns)).issuperset(table_payload_cols):
                        table_access_paths.append({'scan_type': 'Index Only Scan', 'index_id': index.index_id})
                        if verbose: print("Index only scan possible!")

            access_paths[table_name] = table_access_paths

        if verbose:
            print(f"\nEnumerated feasible access paths: ")
            for table, paths in access_paths.items():
                print(f"Table: {table}")
                for path in paths:
                    print(f"    {path}") 

        return access_paths, predicates        



    def calculate_row_overhead(self, num_nullable_columns=0):
        # Tuple header size
        tuple_header_size = 23  # bytes
        # Null bitmap size (1 byte for every 8 nullable columns)
        null_bitmap_size = (num_nullable_columns + 7) // 8
        # Total overhead
        total_overhead = tuple_header_size + null_bitmap_size

        return total_overhead


    def table_avg_rows_per_page(self, table_stats_dict):
        # add up the average width of all columns to get the average width of a row
        avg_row_size = 0
        avg_row_size = sum(column_stats['avg_width'] for column_stats in table_stats_dict.values())
        # add the row overhead
        avg_row_size += self.calculate_row_overhead()
        # calculate the average number of rows that can fit in a page
        avg_rows_per_page = int(get_page_size() / avg_row_size)

        return avg_rows_per_page


    def index_average_rows_per_page(self, index, table_stats_dict):
        columns = list(index.index_columns) + list(index.include_columns)   
        # add up the average width of all columns to get the average width of a row
        avg_row_size = sum(table_stats_dict[column]['avg_width'] for column in columns)
        # add the row overhead
        index_row_overhead = 16  # assume 16 bytes 
        avg_row_size += index_row_overhead
        # calculate the average number of rows that can fit in a page
        # (assuming the index is a B+ tree, so only the leaf nodes contain the actual data)
        avg_rows_per_page = int(get_page_size() / avg_row_size)
        
        return avg_rows_per_page


    def estimate_index_scan_cost(self, index, table_stats_dict, table_predicates, total_rows, cost_multiplier=4.0, index_only_scan=False, verbose=False):
        # check if leading index column is in the predicates
        leading_index_column = index.index_columns[0]
        #print(f"\t\t\tTable predicates: {table_predicates}, Leading index column: {leading_index_column}")
        predicate_columns = [self.id2predicate[pred_id]['column'] for pred_id in table_predicates]
        join_columns = [self.id2predicate[pred_id]['column'] for pred_id in table_predicates if self.id2predicate[pred_id]['join'] == True]

        if leading_index_column not in predicate_columns:
            # assign high cost to prevent using this index if the leading index column is not in the predicates, sequential scan will be cheaper
            return float('inf')
        
        # calculate the combined selectivity for this index (assuming attribute independence/no correlations of predicates)
        leading_column_selectivity = 1.0
        combined_selectivity = 1.0
        for pred_id in table_predicates:
            pred = self.id2predicate[pred_id]
            if pred['column'] in index.index_columns and pred['join'] == False:
                selectivity = self.estimate_selectivity(pred['column'], pred['operator'], pred['value'], table_stats_dict, total_rows)
                if verbose: print(f"\t\tSelectivity for predicate {pred}: {selectivity}")
                combined_selectivity *= selectivity
                if pred['column'] == leading_index_column:
                    leading_column_selectivity = selectivity

        # estimate cardinality of the index scan
        index_cardinality = leading_column_selectivity * total_rows
        # estimate the number of pages that need to be accessed
        avg_rows_per_page = self.index_average_rows_per_page(index, table_stats_dict)
        index_pages = max(1, int(index_cardinality / avg_rows_per_page))
        
        table_pages = 0
        if not index_only_scan: 
            table_cardinality = combined_selectivity * total_rows
            # for index scan, we need to access the table as well
            index_average_rows_per_page_table = self.table_avg_rows_per_page(table_stats_dict)
            table_pages = max(1, int(table_cardinality / index_average_rows_per_page_table))
        

        # apply discount factor to scan cost for join columns being present in the index key or included columns
        # (this is to encourage the use of indexes that cover join columns)
        discount_factor = 1.0
        # add higher discount factor if leading columns is in join columns
        if leading_index_column in join_columns:
            discount_factor *= 0.8
        # add lower discount factor if other key or include columns are in join columns
        other_columns = list(index.index_columns[1:]) + list(index.include_columns)
        for column in other_columns:
            if column in join_columns:
                discount_factor *= 0.9

        # return total cost as the sum of index and table pages
        undiscounted_index_scan_cost = cost_multiplier * (index_pages + table_pages) 
        discounted_index_scan_cost = discount_factor * undiscounted_index_scan_cost
        
        if verbose: 
            print(f"\tLeading column: {leading_index_column}, Predicate columns: {predicate_columns}, Join columns: {join_columns}")
            print(f"\tLeading column selectivity: {leading_column_selectivity}, Combined selectivity: {combined_selectivity}")
            print(f"\tEstimated number of pages for index scan: {index_pages}, Table pages: {table_pages}")
            print(f"\tUndiscounted scan cost --> {undiscounted_index_scan_cost}, Discounted scan cost --> {discounted_index_scan_cost}, Discount factor --> {discount_factor}")

        return discounted_index_scan_cost


    def estimate_sequentail_scan_cost(self, table_stats_dict, total_rows, cost_multiplier=1.0, verbose=False):
        # estimate cardinality of the scan
        scan_cardinality = total_rows
        # estimate the number of pages that need to be accessed
        avg_rows_per_page = self.table_avg_rows_per_page(table_stats_dict)
        scan_pages = int(scan_cardinality / avg_rows_per_page)
        # estimate the total cost as the number of pages that need to be accessed
        sequential_scan_cost = scan_pages * cost_multiplier

        if verbose: 
            print(f"\tEstimated number of pages for sequential scan: {scan_pages}")
            print(f"\tSequential scan cost: {sequential_scan_cost}")

        return sequential_scan_cost


    def find_cheapest_paths(self, access_paths, predicates, indexes, stats, estimated_rows, sequential_scan_cost_multiplier=1.0, index_scan_cost_multiplier=3.0, verbose=False):
        cheapest_table_access_path = {}    
        if verbose: print(f"Finding cheapest access paths for tables: {access_paths.keys()}")
        # enumerate over tables that need to be accessed
        for table_name in access_paths:
            if verbose: print(f"\nTable: {table_name}")
            # enumerate over access paths for this table
            cheapest_cost = float('inf')
            for path in access_paths[table_name]:
                if verbose: print(f"\tComputing cost for access path: {path}")
                # compute the cost of this access path
                # (for now, assume cost is proportional to the cardinality of the data that needs to be accessed)
                if path['scan_type'] == 'Sequential Scan':
                    cost = self.estimate_sequentail_scan_cost(stats[table_name], estimated_rows[table_name], cost_multiplier=sequential_scan_cost_multiplier, verbose=verbose)
                elif path['scan_type'] == 'Index Scan':
                    index_id = path['index_id']
                    index = indexes[index_id]
                    cost = self.estimate_index_scan_cost(index, stats[table_name], predicates[table_name], estimated_rows[table_name], cost_multiplier=index_scan_cost_multiplier, verbose=verbose)    
                elif path['scan_type'] == 'Index Only Scan':
                    index_id = path['index_id']
                    index = indexes[index_id]
                    cost = self.estimate_index_scan_cost(index, stats[table_name], predicates[table_name], estimated_rows[table_name], cost_multiplier=index_scan_cost_multiplier, index_only_scan=True, verbose=verbose)
                else:
                    raise ValueError("Scan type not supported")            

                if verbose: print(f"\tAccess path: {path}, Cost: {cost}\n")
                if cost < cheapest_cost:
                    cheapest_cost = cost
                    cheapest_access_path = path

            cheapest_table_access_path[table_name] = (cheapest_access_path, cheapest_cost)
            if verbose: print(f"\tCheapest access path: {cheapest_access_path}, Cost: {cheapest_cost}")
        
        return cheapest_table_access_path













