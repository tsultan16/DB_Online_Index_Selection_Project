""" 
    Library of miscellaneous helper functions for communicating with MS SQL server
"""


import logging
import datetime
import os
import subprocess
import uuid

import pyodbc
import sys
import random
import pandas as pd
import time
import os
import logging
import re
import json
import xml.etree.ElementTree as ET



""" Code originally from Malinga Perera's work """
class QueryPlan:
    def __init__(self, xml_string):
        self.estimated_rows = 0
        self.est_statement_sub_tree_cost = 0
        self.elapsed_time = 0
        self.cpu_time = 0
        self.non_clustered_index_usage = []
        self.clustered_index_usage = []

        ns = {'sp': 'http://schemas.microsoft.com/sqlserver/2004/07/showplan'}
        root = ET.fromstring(xml_string)
        stmt_simple = root.find('.//sp:StmtSimple', ns)
        if stmt_simple is not None:
            self.estimated_rows = float(stmt_simple.attrib.get('StatementEstRows', 0))
            self.est_statement_sub_tree_cost = float(stmt_simple.attrib.get('StatementSubTreeCost', 0))

        query_stats = root.find('.//sp:QueryTimeStats', ns)
        if query_stats is not None:
            self.cpu_time = float(query_stats.attrib.get('CpuTime', 0))
            self.elapsed_time = float(query_stats.attrib.get('ElapsedTime', 0)) / 1000

        rel_ops = root.findall('.//sp:RelOp', ns)
        total_po_sub_tree_cost = 0
        total_po_actual = 0

        for rel_op in rel_ops:
            temp_act_elapsed_time = 0
            if rel_op.attrib.get('PhysicalOp') in {'Index Seek', 'Index Scan', 'Clustered Index Scan', 'Clustered Index Seek'}:
                total_po_sub_tree_cost += float(rel_op.attrib.get('EstimatedTotalSubtreeCost', 0))
                runtime_thread_information = rel_op.findall('.//sp:RunTimeCountersPerThread', ns)
                for thread_info in runtime_thread_information:
                    temp_act_elapsed_time = max(
                        int(thread_info.attrib.get('ActualElapsedms', 0)), temp_act_elapsed_time)
                total_po_actual += temp_act_elapsed_time / 1000

        for rel_op in rel_ops:
            rows_read = 0
            act_rel_op_elapsed_time = 0
            if rel_op.attrib.get('PhysicalOp') in {'Index Seek', 'Index Scan', 'Clustered Index Scan', 'Clustered Index Seek'}:
                runtime_thread_information = rel_op.findall('.//sp:RunTimeCountersPerThread', ns)
                for thread_info in runtime_thread_information:
                    rows_read += int(thread_info.attrib.get('ActualRowsRead', 0))
                    act_rel_op_elapsed_time = max(int(thread_info.attrib.get('ActualElapsedms', 0)), act_rel_op_elapsed_time)
            act_rel_op_elapsed_time = act_rel_op_elapsed_time / 1000
            if rows_read == 0:
                rows_read = float(rel_op.attrib.get('EstimatedRowsRead', 0))
            rows_output = float(rel_op.attrib.get('EstimateRows', 0))
            if rel_op.attrib.get('PhysicalOp') in {'Index Seek', 'Index Scan'}:
                po_index_scan = rel_op.find('.//sp:IndexScan', ns)
                if po_index_scan is not None:
                    po_index = po_index_scan.find('.//sp:Object', ns).attrib.get('Index', '').strip("[]")
                    self.non_clustered_index_usage.append(
                        (po_index, act_rel_op_elapsed_time, self.cpu_time, self.est_statement_sub_tree_cost, rows_read, rows_output))
            elif rel_op.attrib.get('PhysicalOp') in {'Clustered Index Scan', 'Clustered Index Seek'}:
                po_index_scan = rel_op.find('.//sp:IndexScan', ns)
                if po_index_scan is not None:
                    table = po_index_scan.find('.//sp:Object', ns).attrib.get('Table', '').strip("[]")
                    self.clustered_index_usage.append(
                        (table, act_rel_op_elapsed_time, self.cpu_time, self.est_statement_sub_tree_cost, rows_read, rows_output))
                    


def start_connection(server="172.16.6.196,1433", database="TPCH1", username="wsl", password="greatpond501"):
    conn_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    try:
        connection = pyodbc.connect(conn_string)
    except pyodbc.Error as e:
        logging.error(f"Error connecting to database: {e}")
        sys.exit(1)
    return connection


def close_connection(connection):
    connection.close()


def execute_simple(query, connection):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
    except pyodbc.Error as e:
        print(f"Error executing query {query}: {e}")
    finally:
        cursor.close()    


def execute_query(query, connection, cost_type='elapsed_time', verbose=False):
    cursor = connection.cursor()
    try:
        # clear cache
        cursor.execute("DBCC DROPCLEANBUFFERS")
        # enable statistics collection
        cursor.execute("SET STATISTICS XML ON")
        # execute the query
        cursor.execute(query)
        cursor.nextset()
        # fetch execution stats
        stat_xml = cursor.fetchone()[0]
        cursor.execute("SET STATISTICS XML OFF")
        # parse query plan
        query_plan = QueryPlan(stat_xml)

        if verbose:
            print(f"QUERY: \n{query}\n")
            print(f"ELAPSED TIME: \n{query_plan.elapsed_time}\n")
            print(f"CPU TIME: \n{query_plan.cpu_time}\n")
            print(f"SUBTREE COST: \n{query_plan.est_statement_sub_tree_cost}\n")
            print(f"NON CLUSTERED INDEX USAGE: \n{query_plan.non_clustered_index_usage}\n")
            print(f"CLUSTERED INDEX USAGE: \n{query_plan.clustered_index_usage}\n")
        
    except pyodbc.Error as e:
        logging.error(f"Error executing query: {query}, Error: {e}")
        return 0, [], []
    finally:
        cursor.close()

    # Determine the cost type and return the appropriate metric
    if cost_type == 'elapsed_time':
        return float(query_plan.elapsed_time), query_plan.non_clustered_index_usage, query_plan.clustered_index_usage
    elif cost_type == 'cpu_time':
        return float(query_plan.cpu_time), query_plan.non_clustered_index_usage, query_plan.clustered_index_usage
    elif cost_type == 'sub_tree_cost':
        return float(query_plan.est_statement_sub_tree_cost), query_plan.non_clustered_index_usage, query_plan.clustered_index_usage
    else:
        return float(query_plan.est_statement_sub_tree_cost), query_plan.non_clustered_index_usage, query_plan.clustered_index_usage


def create_nonclustered_index_query(query, connection, verbose=False):
    cursor = connection.cursor()
    try:
        cursor.execute("SET STATISTICS XML ON")
        cursor.execute(query)
        stat_xml = cursor.fetchone()[0]
        cursor.execute("SET STATISTICS XML OFF")
        connection.commit()    

        if verbose:
            #print(f"Query: {query}")
            # Extract the index name
            index_start = query.upper().find("CREATE NONCLUSTERED INDEX") + len("CREATE NONCLUSTERED INDEX")
            index_end = query.upper().find("ON", index_start)
            index_name = query[index_start:index_end].strip()

            # Extract the table name
            table_start = query.upper().find("ON", index_end) + len("ON")
            table_end = query.find("(", table_start)
            table_name = query[table_start:table_end].strip()

            # Extract the indexed columns
            columns_start = query.find("(", table_end) + 1
            columns_end = query.find(")", columns_start)
            indexed_columns = [col.split()[0].strip() for col in query[columns_start:columns_end].split(",")]

            # Extract the included columns
            include_start = query.upper().find("INCLUDE", columns_end)
            if include_start != -1:
                include_start = query.find("(", include_start) + 1
                include_end = query.find(")", include_start)
                included_columns = [col.strip() for col in query[include_start:include_end].split(",")]
            else:
                included_columns = []
            
            print(f"Created index --> {table_name}.{index_name}, Indexed Columns --> {indexed_columns}, Included Columns --> {included_columns}")

            # get index creation time
        query_plan = QueryPlan(stat_xml)
        elapsed_time = query_plan.elapsed_time
        #cpu_time = query_plan.cpu_time

    except pyodbc.Error as e:
        print(f"Error creating index {query}: {e}")
        elapsed_time = 0
    finally:
        cursor.close()    

    return elapsed_time


def drop_nonclustered_index(connection, schema_name=None, table_name=None, index_name=None, query=None, verbose=False):
    cursor = connection.cursor()
    if query is None:
        query = f"DROP INDEX {schema_name}.{table_name}.{index_name}"
    else:
        # extract the schema, table and index names from the query
        split = query.split()
        index_name = split[2][1:-1]
        schema_name = split[4].split('.')[0][1:-1]
        table_name = split[4].split('.')[1][1:-1]
    try:
        cursor.execute(query)
        connection.commit()
        if verbose:
            print(f"Dropped index --> [{schema_name}].[{table_name}].[{index_name}]")
    except pyodbc.Error as e:
        print(f"Error dropping index [{schema_name}].[{table_name}].[{index_name}]: {e}")     
    finally:
        cursor.close()          


def get_nonclustered_indexes(connection):
    cursor = connection.cursor()
    query = """
            SELECT 
            s.name AS SchemaName,
            t.name AS TableName,
            i.name AS IndexName
            FROM 
                sys.indexes i
            JOIN 
                sys.tables t ON i.object_id = t.object_id
            JOIN 
                sys.schemas s ON t.schema_id = s.schema_id
            WHERE 
                i.type_desc = 'NONCLUSTERED'  -- Only non-clustered indexes
                AND i.is_primary_key = 0  -- Exclude primary key indexes
                AND i.is_unique_constraint = 0  -- Exclude unique constraints
            ORDER BY 
                s.name, t.name, i.name; 
            """
    try:
        cursor.execute(query)
        indexes = cursor.fetchall() # return list of tuples: (schema_name, table_name, index_name)
    except pyodbc.Error as e:
        print(f"Error fetching non-clustered indexes: {e}")
        indexes = []
    finally:    
        cursor.close()
    
    return indexes


def remove_all_nonclustered_indexes(connection, verbose=False):
    # get all non-clustered indexes
    indexes = get_nonclustered_indexes()
    print(f"All non-clustered indexes --> {indexes}")
    # drop all non-clustered indexes
    for (schema_name, table_name, index_name) in indexes:
        drop_nonclustered_index(connection, schema_name=schema_name, table_name=table_name, index_name=index_name)

    if verbose:
        print("All nonclustered indexes removed.")


# get size of all PDS in the database
def get_current_pds_size(connection):
    cursor = connection.cursor()
    query = '''SELECT (SUM(s.[used_page_count]) * 8)/1024.0 AS size_mb FROM sys.dm_db_partition_stats AS s'''
    try:
        cursor.execute(query)
        pds_size = cursor.fetchone()[0]
    except pyodbc.Error as e:
        print(f"Error fetching PDS size: {e}")
        pds_size = 0
    finally:
        cursor.close()    

    return pds_size


