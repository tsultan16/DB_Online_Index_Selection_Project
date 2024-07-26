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
import copy
from collections import defaultdict


tables_global = None
pk_columns_dict = {}
MAX_TABLE_SCAN_TIME_SAMPLES = 1000


class Query:
    def __init__(self, connection, template_id, query_string, payload, predicates, order_by, group_by, time_created=0, benchmark="TPC-H"):
        self.template_id = template_id
        self.query_string = query_string
        self.payload = payload
        self.predicates = predicates
        self.order_by = order_by
        self.group_by = group_by
        self.benchmark = benchmark
        self.selectivity = estimate_selectivity(connection, query_string, predicates)
        self.frequency = 1
        self.first_seen = time_created
        self.last_seen = time_created
        self.table_scan_times = get_query_table_scan_times(connection, query_string, predicates)
        self.index_scan_times = copy.deepcopy(self.table_scan_times)
        self.context_vector = None

    def get_id(self):
        return self.template_id

    def __hash__(self):
        return self.template_id

    def __str__(self):
        return f"template: {self.template_id}\n\query string: {self.query_string}\npayload: {self.payload}\npredicates: {self.predicates}\norder_by: {self.order_by}"



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


def get_database_size(connection):
    database_size = 10240
    try:
        query = "exec sp_spaceused @oneresultset = 1;"
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        database_size = float(result[4].split(" ")[0])/1024
    except Exception as e:
        logging.error("Exception when get_database_size: " + str(e))
    return database_size



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
        # clear out the buffer pool for cold start
        cursor.execute("DBCC DROPCLEANBUFFERS")
        # Clear the procedure/query optimizer's cache
        #cursor.execute("DBCC FREEPROCCACHE")
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


def bulk_execute_queries(queries, connection, cost_type='elapsed_time', verbose=False):
    total_cost = 0
    for query in queries:
        elapsed_time, non_clustered_index_usage, clustered_index_usage = execute_query(query, connection, cost_type=cost_type, verbose=verbose)
        total_cost += elapsed_time
    return total_cost


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
    indexes = get_nonclustered_indexes(connection)
    #print(f"All non-clustered indexes --> {indexes}")
    # drop all non-clustered indexes
    for (schema_name, table_name, index_name) in indexes:
        drop_nonclustered_index(connection, schema_name=schema_name, table_name=table_name, index_name=index_name)

    print(f"All {len(indexes)} nonclustered indexes removed.")


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


# get XML query plan for a given query
def get_query_plan(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute("SET SHOWPLAN_XML ON")
        cursor.execute(query)
        plan_xml = cursor.fetchone()[0]
        cursor.execute("SET SHOWPLAN_XML OFF")
    except pyodbc.Error as e:
        print(f"Error fetching query plan for query {query}: {e}")
        plan_xml = None
    finally:
        cursor.close()    

    return plan_xml


# estimate the selectivity of a query, i.e. the fraction of rows from each table that satisfy the predicates so will be returned by the query
def estimate_selectivity(connection, query, predicates, verbose=False):
    # get query plan xml
    plan_xml = get_query_plan(connection, query)
    if plan_xml != "":
        read_rows = {} 
        selectivity = {}
        query_plan = QueryPlan(plan_xml)
        tables = predicates.keys()
        if verbose: print(f"Tables: {tables}")
        # get num rows scanned for each table scan
        for table in tables:
            read_rows[table] = 1000000000 # initialize to a large number, this is the cap for the number of rows scanned
        
        # look at the clustered index usage to get the number of rows scanned for each table
        for index_scan in query_plan.clustered_index_usage:
            if verbose:
                print(f"Index Scan: {index_scan}")

            if index_scan[0] not in read_rows:
                read_rows[index_scan[0]] = read_rows[table]
            else:
                read_rows[index_scan[0]] = min(index_scan[5], read_rows[index_scan[0]])    

        for table in tables:
            table_row_count = get_table_row_count(connection, 'dbo', table)
            selectivity[table] = min(1.0, read_rows[table] / table_row_count)
            if verbose:
                print(f"Table: {table}, Rows Scanned: {read_rows[table]}, table row count: {table_row_count}, selectivity: {selectivity[table]}")  
 

    else:
        # if query plan is not available, assume selectivity is 1, i.e. full table scan
        selectivity = 1

    return selectivity



# execute a query and get the time taken to scan each table
def get_query_table_scan_times(connection, query_string, predicates):
    # execute the query and get the time taken to scan each table
    table_scan_times = {}
    elapsed_time, index_seeks, clustered_index_scans = execute_query(query_string, connection)
    if clustered_index_scans:
        for index_scan in clustered_index_scans:
            table_name = index_scan[0]
            if table_name not in table_scan_times:
                table_scan_times[table_name] = []
            # add the time taken to scan the table to the list of scan times (upto MAX_TABLE_SCAN_TIME_SAMPLES samples)
            if len(table_scan_times[table_name]) < MAX_TABLE_SCAN_TIME_SAMPLES:
                table_scan_times[table_name].append(index_scan[1])    

    return table_scan_times


# display table as pandas dataframe
def display_table(connection, table, max_tuples=10):
    # execute select query to retreive first max_tuples rows from the table
    table_name = table.table_name
    cursor = connection.cursor()
    query = f"SELECT TOP {max_tuples} * FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [c.column_name for _, c in table.columns.items()]
    df = pd.DataFrame.from_records(rows, columns=columns)
    
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)        # Set display width to None to auto-detect
    print(df)
    
    cursor.close()


# table object
class Table:
    def __init__(self, table_name, row_count, pk_columns):
        self.table_name = table_name
        self.row_count = row_count
        self.pk_columns = pk_columns
        self.columns = None

    def set_columns(self, columns):
        self.columns = columns

    def get_columns(self):  
        if self.columns is None:
            raise Exception("Columns have not been set for this table yet.")                
        return self.columns

    def __str__(self):
        return f"Table: {self.table_name}, Row Count: {self.row_count}, PK Columns: {self.pk_columns}"

# column object
class Column:
    def __init__(self, table_name, column_name, column_type):
        self.table_name = table_name    
        self.column_name = column_name
        self.column_type = column_type
        self.column_size = None
        self.max_column_size = None

    def set_column_size(self, size):
        self.column_size = size

    def set_max_column_size(self, max_size):
        self.max_column_size = max_size

    def get_id(self):
        return f"{self.table_name}_{self.column_name}"    
    
    @staticmethod
    def construct_id(table_name, column_name):
        return f"{table_name}_{column_name}"
        
    def __str__(self):
        return f"Column: {self.column_name}, Column Type Type: {self.column_type}, Column Size: {self.column_size}, Max Column Size: {self.max_column_size}"

# index object
class Index:
    def __init__(self, table_name, index_id, index_columns, size, include_columns=(), value=None):
        self.table_name = table_name
        self.index_id = index_id
        self.index_columns = index_columns
        self.size = size
        self.include_columns = include_columns
        self.value = value

    def __str__(self):
        return f"Index({self.table_name}, {self.index_id}, {self.index_columns}, {self.include_columns}, {self.size}, {self.value})"

# get number of rows in a given table
def get_table_row_count(connection, schema_name, table_name):
    cursor = connection.cursor()
    query = f""" 
                SELECT SUM (Rows)
                FROM sys.partitions
                WHERE index_id IN (0, 1) 
                AND object_id = OBJECT_ID('{schema_name}.{table_name}')
            """
    try:
        cursor.execute(query)
        row_count = cursor.fetchone()[0]
    except pyodbc.Error as e:
        print(f"Error fetching row count for table {table_name}: {e}")
        row_count = 0
    finally:
        cursor.close()    

    return row_count


# get primary key columns of a given table
def get_primary_key_columns(connection, schema_name, table_name):
    if table_name in pk_columns_dict:
        pk_columns = pk_columns_dict[table_name]

    else:
        cursor = connection.cursor()
        query = f"""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
                AND TABLE_NAME = '{table_name}'
                AND TABLE_SCHEMA = '{schema_name}'
                """
        try:
            cursor.execute(query)
            pk_columns = [result[0] for result in cursor.fetchall()]
            pk_columns_dict[table_name] = pk_columns

        except pyodbc.Error as e:
            print(f"Error fetching primary key columns for table {table_name}: {e}")
            pk_columns = []
        finally:
            cursor.close()    

    return pk_columns


# get list of all columns in a given table
def get_table_columns(connection, schema_name, table_name):
    columns = {}
    cursor = connection.cursor()
    try:
        data_type_query = f"""
                            SELECT COLUMN_NAME, DATA_TYPE, COL_LENGTH('{schema_name}.{table_name}', COLUMN_NAME)
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_NAME = '{table_name}'
                            """    
        cursor.execute(data_type_query)
        results = cursor.fetchall()
        variable_len_query = 'SELECT '
        variable_len_select_segments = []
        variable_len_inner_segments = []
        varchar_ids = []
        for result in results:
            col_name = result[0]
            column = Column(table_name, col_name, result[1])
            column.set_max_column_size(int(result[2]))
            if result[1] != 'varchar':
                column.set_column_size(int(result[2]))
            else:
                varchar_ids.append(col_name)
                variable_len_select_segments.append(f'''AVG(DL_{col_name})''')
                variable_len_inner_segments.append(f'''DATALENGTH({col_name}) DL_{col_name}''')
            columns[col_name] = column

        if len(varchar_ids) > 0:
            variable_len_query = variable_len_query + ', '.join(
                variable_len_select_segments) + ' FROM (SELECT TOP (1000) ' + ', '.join(
                variable_len_inner_segments) + f' FROM {table_name}) T'
            cursor.execute(variable_len_query)
            result_row = cursor.fetchone()
            for i in range(0, len(result_row)):
                columns[varchar_ids[i]].set_column_size(result_row[i])
    
    except pyodbc.Error as e:
        print(f"Error fetching columns for table {table_name}: {e}")
    finally:
        cursor.close()    

    return columns


# get list of all tables in the database
def get_all_tables(connection, schema_name='dbo'):
    global tables_global
    if tables_global is not None:
        return tables_global
    
    else:    
        tables = {}
        cursor = connection.cursor()
        query = """
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
                """
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            for result in results:
                table_name = result[0]
                row_count = get_table_row_count(connection, schema_name, table_name)
                pk_columns = get_primary_key_columns(connection, schema_name, table_name)
                # cache the table object
                tables[table_name] = Table(table_name, row_count, pk_columns)
                tables[table_name].set_columns(get_table_columns(connection, schema_name, table_name))

        except pyodbc.Error as e:
            print(f"Error fetching tables: {e}")
        finally:
            cursor.close()

        tables_global = tables    
        return tables
    

# get list of all columns in the database
def get_all_columns(connection):
    cursor = connection.cursor()
    query = """     
            SELECT TABLE_NAME, COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS;
            """
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        columns = defaultdict(list)
        for result in results:
            columns[result[0]].append(result[1])
        num_columns = len(results)        

    except pyodbc.Error as e:
        print(f"Error fetching columns: {e}")
        columns = None
        num_columns = 0
    finally:
        cursor.close()

    return columns, num_columns


# assign a unique id to a given index
def get_index_id(index_cols, table_name, include_cols=()):
    if include_cols:
        include_col_names = '_'.join(tuple(map(lambda x: x[0:4], include_cols))).lower()
        index_id = 'IXN_' + table_name + '_' + '_'.join(index_cols).lower() + '_' + include_col_names
    else:
        index_id = 'IX_' + table_name + '_' + '_'.join(index_cols).lower()
    
    # truncate to 128 chars
    index_id = index_id[:127]
    return index_id


# estimate the size of an index on a given table
def get_estimated_index_size(connection, table_name, columns, schema_name='dbo'):
    # get table object
    table = get_all_tables(connection)[table_name]
    # get primary key columns
    pk_columns = get_primary_key_columns(connection, schema_name, table_name)
    # get size of primary key columns
    pk_size = get_column_data_size(connection, table_name, pk_columns)
    # get non primary key columns and calculate their sizes
    col_not_pk = tuple(set(columns) - set(pk_columns))
    col_not_pk_sizes = get_column_data_size(connection, table_name, col_not_pk)
    # calculate size of each row in bytes
    header_size_bytes = 6
    nullable_buffer_bytes = 1
    index_row_size = header_size_bytes + pk_size + col_not_pk_sizes + nullable_buffer_bytes
    # get row count of table
    row_count = table.row_count
    # get estimated size of index in bytes
    estimated_size = row_count * index_row_size
    # convert to MB
    estimated_size = estimated_size/float(1024*1024)
    max_column_size = get_max_column_data_size(connection, table_name, columns)
    # check if index size is going past 1700 MB
    if max_column_size > 1700:
        logging.info(f'Index size going past 1700 MB: {columns}')
        estimated_size = 99999999
    logging.debug(f"{columns} : {estimated_size}")

    return estimated_size


# compute column data size
def get_column_data_size(connection, table_name, columns):
    tables = get_all_tables(connection)
    varchar_count = 0
    column_data_size = 0
    for column_name in columns:
        column = tables[table_name].columns[column_name]
        if column.column_type == 'varchar':
            varchar_count += 1
        column_data_size += column.column_size if column.column_size else 0
        
    if varchar_count > 0:
        variable_key_overhead = 2 + 2 * varchar_count
        return column_data_size + variable_key_overhead
    else:
        return column_data_size    
    

# compute maximum possible column data size
def get_max_column_data_size(connection, table_name, columns):
    tables = get_all_tables(connection)
    column_data_size = 0
    for column_name in columns:
        column = tables[table_name].columns[column_name]
        column_data_size += column.max_column_size if column.max_column_size else 0
    return column_data_size


# create a non-clustered index for given index object
def create_nonclustered_index_object(connection, index, schema_name='dbo', cost_type='elapsed_time', verbose=False):
    index_id = index.index_id
    table_name = index.table_name
    index_columns = index.index_columns
    include_columns = index.include_columns 

    if include_columns:
        query = f"""
                CREATE NONCLUSTERED INDEX {index_id}
                ON {schema_name}.{table_name} ({', '.join(index_columns)})
                INCLUDE ({', '.join(include_columns)})
                """
    else:
        query = f"""
                CREATE NONCLUSTERED INDEX {index_id}
                ON {schema_name}.{table_name} ({', '.join(index_columns)})
                """

    cursor = connection.cursor()
    try:
        cursor.execute("SET STATISTICS XML ON")
        cursor.execute(query)
        stat_xml = cursor.fetchone()[0]
        cursor.execute("SET STATISTICS XML OFF")
        connection.commit()
        query_plan = QueryPlan(stat_xml)
        if cost_type == 'elapsed_time':
            cost = query_plan.elapsed_time
        elif cost_type == 'cpu_time':
            cost = query_plan.cpu_time
        else:
            cost = query_plan.est_statement_sub_tree_cost
        if verbose:
            print(f"Created index --> [{schema_name}].[{table_name}].[{index_id}], Indexed Columns --> {index_columns}, Included Columns --> {include_columns}, index creation time: {cost} seconds")

    except pyodbc.Error as e:
        print(f"Error creating index {index_id}: {e}")
        cost = 0
    finally:
        cursor.close() 

    return cost           


# drop non-clustered index for given index object
def drop_noncluster_index_object(connection, index, schema_name='dbo', verbose=False):
    index_id = index.index_id
    table_name = index.table_name
    query = f"DROP INDEX {schema_name}.{table_name}.{index_id}"
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        if verbose:
            print(f"Dropped index --> [{schema_name}].[{table_name}].[{index_id}]")
    except pyodbc.Error as e:
        print(f"Error dropping index {index_id}: {e}")
    finally:
        cursor.close()


# bulk create and drop non-clustered indexes
def bulk_create_drop_nonclustered_indexes(connection, indexes_to_create, indexes_to_drop, schema_name='dbo', cost_type='elapsed_time', verbose=False):
    if verbose:
        print(f"Implementing configuration change: Creating {len(indexes_to_create)} indexes and dropping {len(indexes_to_drop)} indexes")
    
    # create indexes
    creation_cost = {}
    for index in indexes_to_create:
        creation_cost[index.index_id] = create_nonclustered_index_object(connection, index, schema_name=schema_name, cost_type=cost_type, verbose=verbose)
    # drop indexes
    for index in indexes_to_drop:
        drop_noncluster_index_object(connection, index, schema_name=schema_name, verbose=verbose)

    if verbose:
        print(f"Total index creation cost: {sum(creation_cost.values())}")

    return creation_cost


"""
    Hypothetical Index Creation and Query Execution Cost Estimation
"""

# create hypothetical indexes for a given set of indexes
def create_hypothetical_indexes(connection, indexes, schema_name='dbo', verbose=False):
    est_index_creation_cost = None

    cursor = connection.cursor()
    for index in indexes:
        try:
            index_id = index.index_id
            table_name = index.table_name
            index_columns = index.index_columns
            include_columns = index.include_columns

            # use the 'WITH STATISTICS_ONLY' option to create hypothetical indexes
            if include_columns:
                query = f"CREATE NONCLUSTERED INDEX {index_id} ON {schema_name}.{table_name} ({', '.join(index_columns)}) " \
                        f"INCLUDE ({', '.join(include_columns)}) WITH STATISTICS_ONLY = -1"
            else:
                query = f"CREATE NONCLUSTERED INDEX {index_id} ON {schema_name}.{table_name} ({', '.join(index_columns)}) " \
                        f"WITH STATISTICS_ONLY = -1"

            cursor.execute(query)
            connection.commit()
            if verbose:
                print(f"Created hypothetical index --> [{table_name}].[{index_id}]")

            # Note: cost estimation currently not implemented...


        except Exception as e:
            print(f"Error creating hypothetical index {index_id}: {e}")
            # Continue to the next index if an error occurs
            continue

    cursor.close()
    return est_index_creation_cost


# drop all hypothetical indexes in the database
def drop_all_hypothetical_indexes(connection, verbose=False):
    cursor = connection.cursor()
    try:
        # first identify all hypothetical indexes
        query = """
                SELECT 
                    OBJECT_SCHEMA_NAME(object_id) AS SchemaName,
                    OBJECT_NAME(object_id) AS TableName,
                    name AS IndexName
                FROM 
                    sys.indexes
                WHERE 
                    is_hypothetical = 1;
                """
        cursor.execute(query)
        hypothetical_indexes = cursor.fetchall()
        
        # drop each hypothetical index
        for index in hypothetical_indexes:
            schema_name = index[0]
            table_name = index[1]
            index_name = index[2]
            drop_query = f"DROP INDEX {schema_name}.{table_name}.{index_name}"
            cursor.execute(drop_query)
            if verbose: 
                print(f"Dropped hypothetical index: {schema_name}.{table_name}.{index_name}")        
        connection.commit()
    
    except Exception as e:
        print(f"Error dropping hypothetical indexes: {e}")
    finally:
        cursor.close()    


# enable all the hypothetical indexes so that they will be considered by optimizer when generating query plans for future queries
def enable_all_hypothetical_indexes(connection, verbose=False):
    query = """ 
                SELECT dbid = Db_id(), objectid = object_id, indid = index_id
                FROM   sys.indexes
                WHERE  is_hypothetical = 1;
            """
    cursor = connection.cursor()
    try:    
        cursor.execute(query)
        result_rows = cursor.fetchall()
        for result_row in result_rows:
            if verbose: print(f"Enabling hypothetical index: {result_row}")
            query_2 = f"DBCC AUTOPILOT(0, {result_row[0]}, {result_row[1]}, {result_row[2]}, 0, 0, 0)"
            cursor.execute(query_2)
        connection.commit()
    except Exception as e:
        print(f"Error enabling hypothetical indexes: {e}")
    finally:
        cursor.close()    


# disable sepcified indexes in the database, these won't be considered by the optimizer when generating query plans for future queries
def disable_indexes(connection, indexes_to_disable, verbose=False):
    try:
        cursor = connection.cursor()
        for index in indexes_to_disable:
            index_id = index.index_id
            table_name = index.table_name
            disable_query = f"ALTER INDEX {index_id} ON {table_name} DISABLE"
            cursor.execute(disable_query)
            if verbose:
                print(f"Disabled index: {index_id}")

        connection.commit()
    except Exception as e:
        print(f"Error disabling indexes: {e}")
    finally:
        cursor.close()    


# re-enable the indexes that were previously disabled    
# ***WARNING*** This will rebuild the indexes from scratch and will be super SLOW
def re_enable_indexes(connection, indexes_to_enable, verbose=False):
    try:
        cursor = connection.cursor()
        for index in indexes_to_enable:
            index_id = index.index_id
            table_name = index.table_name
            enable_query = f"ALTER INDEX {index_id} ON {table_name} REBUILD"
            cursor.execute(enable_query)
            if verbose:
                print(f"Re-enabled index: {index_id}")
        connection.commit()
    except Exception as e:
        print(f"Error enabling indexes: {e}")
    finally:
        cursor.close()    


# extract query plan for a query in the hypothetical configuration to estimate the execution cost
"""
def hypothetical_execute_query(connection, query, verbose=False):
    cursor = connection.cursor()
    try:
        # set AUTOPILOT so that hypothetical indexes are considered
        cursor.execute("SET AUTOPILOT ON")
        cursor.execute("SET SHOWPLAN_XML ON")
        cursor.execute(query)
        stat_xml = cursor.fetchone()[0]
        cursor.execute("SET SHOWPLAN_XML OFF")
        cursor.execute("SET AUTOPILOT OFF")
        query_plan = QueryPlan(stat_xml)
        return float(query_plan.est_statement_sub_tree_cost), query_plan.non_clustered_index_usage, query_plan.clustered_index_usage
    
    except Exception as e:
        print(f"Error executing hypothetical query: {e}")
        return None, None, None
    finally:
        cursor.close()
"""


def hypothetical_execute_query(connection, query, verbose=False):
    cursor = connection.cursor()
    try:
        # Log start time
        start_time = time.time()
        
        # set AUTOPILOT so that hypothetical indexes are considered
        cursor.execute("SET AUTOPILOT ON")
        cursor.execute("SET SHOWPLAN_XML ON")
        
        # Log time after setting AUTOPILOT and SHOWPLAN_XML
        mid_time = time.time()
        
        cursor.execute(query)
        stat_xml = cursor.fetchone()[0]
        
        # Log time after executing the query
        end_time = time.time()
        
        cursor.execute("SET SHOWPLAN_XML OFF")
        cursor.execute("SET AUTOPILOT OFF")
        
        query_plan = QueryPlan(stat_xml)
        
        # Log total time taken
        total_time = end_time - start_time
        setup_time = mid_time - start_time
        execution_time = end_time - mid_time
        
        if verbose:
            print(f"Setup time: {setup_time:.6f} seconds")
            print(f"Execution time: {execution_time:.6f} seconds")
            print(f"Total time: {total_time:.6f} seconds")
        
        return float(query_plan.est_statement_sub_tree_cost), query_plan.non_clustered_index_usage, query_plan.clustered_index_usage
    
    except Exception as e:
        print(f"Error executing hypothetical query: {e}")
        return None, None, None
    finally:
        cursor.close()



# estimate the cost of executing a query in the current configuration from the query plan
def estimate_query_cost(connection, query):
    cursor = connection.cursor()        
    try:
        #cursor.execute("DBCC FREEPROCCACHE") # clear out query plan cache
        cursor.execute("SET SHOWPLAN_XML ON")
        cursor.execute(query)
        stat_xml = cursor.fetchone()[0]
        cursor.execute("SET SHOWPLAN_XML OFF")
        query_plan = QueryPlan(stat_xml)
        return float(query_plan.est_statement_sub_tree_cost), query_plan.non_clustered_index_usage, query_plan.clustered_index_usage   

    except Exception as e:
        print(f"Error estimating query cost: {e}")
        return None, None, None
    finally:
        cursor.close()

# estimate bulk query cost
def bulk_estimate_query_cost(connection, queries):
    total_cost = 0
    for query in queries:
        cost, _, _ = estimate_query_cost(connection, query)
        total_cost += cost
    return total_cost


# estimate the cost of executing a batch of queries in a hypothetical configuration
def hyp_configuration_cost_estimate(connection, indexes_added, indexes_removed, queries, verbose=False):
    
    t1 = time.time()
    # create hypothetical indexes
    hyp_index_creation_cost = create_hypothetical_indexes(connection, indexes_added, verbose=verbose)
    t2 = time.time()
    print(f"Time taken to create hypothetical indexes: {t2-t1}")
    # disable removed indexes
    t1 = time.time()
    disable_indexes(connection, indexes_removed, verbose=verbose)
    t2 = time.time()
    print(f"Time taken to disable indexes: {t2-t1}")

    # execute query in hypothetical configuration
    enable_all_hypothetical_indexes(connection, verbose=verbose)
    total_hyp_cost = 0
    total_orig_cost = 0
    tH = 0
    tO = 0
    for query in queries:
        t1 = time.time()
        hyp_execution_cost, non_clustered_index_usage, clustered_index_usage = hypothetical_execute_query(connection, query, verbose=verbose)
        t2 = time.time()
        tH += (t2-t1)

        t1 = time.time()
        original_execution_cost, non_clustered_index_usage, clustered_index_usage = estimate_query_cost(connection, query)
        t2 = time.time()
        tO += (t2-t1)
        
        total_hyp_cost += hyp_execution_cost
        total_orig_cost += original_execution_cost

    print(f"Time to estimate --> Execution Cost Original: {tO}, Execution Cost Hypothetical: {tO}")

    # drop hypothetical indexes
    t1 = time.time()
    drop_all_hypothetical_indexes(connection, verbose=verbose)
    t2 = time.time()
    print(f"Time taken to drop hypothetical indexes: {t2-t1}")
    
    # re-enable oreviously disabled indexes
    t1 = time.time()
    re_enable_indexes(connection, indexes_removed, verbose=verbose)
    t2 = time.time()
    print(f"Time taken to re-enable indexes: {t2-t1}\n")    

    return total_orig_cost, total_hyp_cost, hyp_index_creation_cost
    

































