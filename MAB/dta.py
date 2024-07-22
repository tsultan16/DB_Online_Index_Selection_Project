""" 
    A simple interface for invoking the DTA Recommender
"""

import os
import sys
import re

script_directory = os.path.dirname(os.path.abspath(__file__))
target_subdirectory_path = os.path.join(script_directory, 'database')
sys.path.append(target_subdirectory_path)
from utils import *



"""
    DTA interface class
"""
class DTA_recommender:
    def __init__(self, invoke_ta_rounds=[], verbose=False):
        self.invoke_ta_rounds = invoke_ta_rounds # list specifying the rounds to invoke TA
        self.verbose = verbose
        self.server="172.16.6.196,1433"
        self.database="TPCH1" 
        self.username="wsl" 
        self.password="greatpond501"
        
    def parse_create_index_query(self, query_string):
        pattern = r"CREATE NONCLUSTERED INDEX \[([^\]]+)\] ON \[([^\]]+)\]\.\[([^\]]+)\]\s*\(([^)]+)\)(?:\s*INCLUDE\s*\(([^)]+)\))?"

        # Search for the pattern in the query string
        match = re.search(pattern, query_string, re.DOTALL)

        if match:
            index_name = match.group(1)
            schema_name = match.group(2)
            table_name = match.group(3)
            index_columns = [col.strip() for col in match.group(4).split(',')]
            include_columns = [col.strip() for col in match.group(5).split(',')] if match.group(5) else []

            index_columns = [column_name.replace('[', '').replace(']', '')for column_name in index_columns]
            include_columns = [column_name.replace('[', '').replace(']', '') for column_name in include_columns]

            #print("Index Name:", index_name)
            #print("Schema Name:", schema_name)
            #print("Table Name:", table_name)
            #print("Index Columns:", index_columns)
            #print("Include Columns:", include_columns if include_columns else [])
            return {'index_name':index_name, 'table_name':table_name, 'index_columns':index_columns, 'include_columns':include_columns}
        else:
            print("No match found. Please check the CREATE INDEX query format.")   
            return {}


    def parse_drop_index_query(self, query_string):
        # Regex pattern to parse the DROP INDEX query
        pattern = r"DROP INDEX IF EXISTS (?:\[(.*?)\]\.)?\[(.*?)\](?: ON \[(.*?)\]\.\[(.*?)\])?;"

        # Search for the pattern in the query string
        match = re.search(pattern, query_string)

        if match:
            schema_name = match.group(1)
            index_name = match.group(2)
            table_schema = match.group(3)  # This might be None if the schema name is not specified in the query
            table_name = match.group(4)    # This might be None if the table name is not specified in the query

            print("Schema Name:", schema_name if schema_name else "Not specified")
            print("Index Name:", index_name)
            print("Table Schema:", table_schema if table_schema else "Not specified")
            print("Table Name:", table_name if table_name else "Not specified")
            return {'index_name':index_name, 'table_name':table_name}
        else:
            print("No match found. Please check the DROP INDEX query format.")    
            return {}


    # run DTA and get the recommended configuration for a single batch of queries
    def recommend_indexes(self, queries, max_memory_Mb=2048, max_time_minutes=1, clear_indexes_start=True):
        self.conn = start_connection()
        # clear all non-clustered indexes at the start
        if clear_indexes_start:   
            remove_all_nonclustered_indexes(self.conn, self.verbose)

        # reset workload file
        self.workload_file = "workload_tpch1.sql"
        open(self.workload_file, 'w').close()
        
        # write the queries to workload file
        with open(self.workload_file, 'a+') as file:
            for query_string in queries:
                # exclude queries with "view" in them, otherwise DTA will throw a syntax error
                if "view" not in query_string.lower():
                    file.write(query_string)
                    file.write('\n\n\n')

        # invoke DTA 
        recommendation_cost, recommendation_output_file = self.get_recommendations(max_memory_Mb, max_time_minutes)      

        try:
            with open(recommendation_output_file, 'r', encoding="utf-16") as file:
                query_lines = file.readlines()
                sql = ' '.join(query_lines)
                sql = sql.replace('go\n', ';')
        except Exception as e:
            print(f"Error reading recommendations file: {e}")
            return 0                    
        
        recommendation_queries = sql.split(';')

        # parse the recommendation output file to get a list of all recommended non-clustered index creations
        indexes_to_add = []
        indexes_to_remove = []
        for query in recommendation_queries[1:]:
            if not query.isspace():
                if "create nonclustered index" in query.lower():
                    indexes_to_add.append(self.parse_create_index_query(query))
                    #indexes_to_add.append(query)
                elif "drop index" in query.lower():
                    indexes_to_remove.append(self.parse_drop_index_query(query))
                    #indexes_to_remove.append(query)

        # reset the workload file
        open(self.workload_file, 'w').close()
        # clear out all the recommendations and session files from directory
        for file in os.listdir():
            if file.startswith("recommendations") or file.startswith("session_output"):
                os.remove(file)

        close_connection(self.conn)
        print(f"DTA recommendation time: {recommendation_cost} seconds.")        

        return indexes_to_add, indexes_to_remove


    # run DTA and implement recommended configurations over specified number of rounds    
    def run_dta(self, queries, num_rounds=1, invoke_DTA=True, clear_indexes_start=False, clear_indexes_end=True):

        # establish connection to the database
        self.conn = start_connection()

        # clear all non-clustered indexes at the start
        if clear_indexes_start:   
            remove_all_nonclustered_indexes(self.conn, self.verbose)
  
        if num_rounds > 0:

            # reset workload file
            self.workload_file = "workload_tpch1.sql"
            open(self.workload_file, 'w').close()

            num_queries_per_round = len(queries) // num_rounds

            # iterate over rounds
            counter = 0
            for i in range(num_rounds):
                print(f"Round {i+1} of {num_rounds}")
                current_round_queries = queries[i*num_queries_per_round:(i+1)*num_queries_per_round]
                
                if invoke_DTA:
                    # write the queries for current round to the workload file
                    with open(self.workload_file, 'a+') as file:
                        for query in current_round_queries:
                            query_string = query['query_string']
                            # exclude queries with "view" in them, otherwise DTA will throw a syntax error
                            if "view" not in query_string.lower():
                                file.write(query_string)
                                file.write('\n\n\n')
                                counter += 1

                    print(f"{counter} queries written to workload file")

                    # invoke DTA if current round is in invoke_ta_rounds
                    if i in self.invoke_ta_rounds:
                        recommendation_cost_round, recommmendation_output_file = self.get_recommendations()      
                        if os.path.isfile(recommmendation_output_file):
                            self.implement_recommendations(recommmendation_output_file)
                            # reset the workload file
                            open(self.workload_file, 'w').close()
                            counter = 0

                # now execute the workload for current round
                execution_cost_round = self.execute_workload(current_round_queries)        

            # clear all indexes
            if clear_indexes_end: remove_all_nonclustered_indexes(self.conn, self.verbose)
            # clear out all the recommendations and session files from directory
            for file in os.listdir():
                if file.startswith("recommendations") or file.startswith("session_output"):
                    os.remove(file)

        # close the connection
        close_connection(self.conn)

        
    def get_recommendations(self, max_memory_Mb=1024, max_time_minutes=1):
        session_name = f"session_{uuid.uuid4()}"
        max_memory = max_memory_Mb 
        max_time = max_time_minutes
        recommendation_output_file = f"recommendations_{session_name}.sql"
        session_output_xml_file = f"session_output_{session_name}.xml"        
        dta_exe_path = '"/mnt/c/Program Files (x86)/Microsoft SQL Server Management Studio 20/Common7/DTA.exe"'
        dta_command = f'{dta_exe_path} -S 172.16.6.196 -U wsl -P greatpond501 -D {self.database} -d {self.database} ' \
                    f'-if "{self.workload_file}" -s {session_name} ' \
                    f'-of "{recommendation_output_file}" ' \
                    f'-ox "{session_output_xml_file}" ' \
                    f'-fa NCL_IDX -fp NONE -fk CL_IDX -B {max_memory} -A {max_time} -F'

        start_time = datetime.datetime.now()
        subprocess.run(dta_command, shell=True)
        end_time = datetime.datetime.now()
        time_elapsed = (end_time - start_time).total_seconds()
        
        print(f"DTA recommendation time --> {time_elapsed} seconds.")
                  
        return time_elapsed, recommendation_output_file     


    def implement_recommendations(self, recommendation_output_file):
        if self.verbose: print("Implementing recommendations...")
        try:
            with open(recommendation_output_file, 'r', encoding="utf-16") as file:
                query_lines = file.readlines()
                sql = ' '.join(query_lines)
                sql = sql.replace('go\n', ';')
        except Exception as e:
            print(f"Error reading recommendations file: {e}")
            return 0                    

        recommendation_queries = sql.split(';')
        #if self.verbose:
        #    print(f"Recommendation queries: \n{recommendation_queries}")
        
        total_index_creation_cost = 0
        for query in recommendation_queries[1:]:
            if not query.isspace():
                if "create nonclustered index" in query.lower():
                    total_index_creation_cost += create_nonclustered_index_query(query, self.conn, verbose=self.verbose) 
                elif "drop index" in query.lower():
                    drop_nonclustered_index(self.conn, query=query, verbose=self.verbose)

        print(f"Implemented recommendations.")
        print(f"Total index creation time --> {total_index_creation_cost} seconds. Total size of configuration --> {get_current_pds_size(self.conn)} MB")

        return total_index_creation_cost


    def execute_workload(self, workload):
        if self.verbose:
            print(f"Executing workload of {len(workload)} queries")
        total_elapsed_time = 0
        # execute the workload
        for query in workload:
            cost, index_seeks, clustered_index_scans = execute_query(query['query_string'], self.conn)
            total_elapsed_time += cost   
        print(f"Current round workload execution time --> {total_elapsed_time} seconds.")     

        return total_elapsed_time




