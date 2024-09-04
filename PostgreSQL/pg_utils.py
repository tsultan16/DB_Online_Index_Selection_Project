"""
    A collection of Util Functions for talking to PostgreSQL database
"""

import psycopg2
import json
import itertools
import time
import subprocess


# restart the PostgreSQL server and clear OS cache
def restart_postgresql(clear_cache=False, delay=2):

    # Path to the shell script
    script_path = '/home/tanzid/Code/Postgres/postgres/restart_cache.sh'

    # Construct the command to run the shell script
    command = [script_path, '-d', str(delay)]
    if clear_cache:
        command.extend(['-c', 'true'])

    try:
        # Run the shell script
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


# create connection to postgres DB
def create_connection():
    connection_params = {
        "dbname": "SSB10",
        "user": "tanzid",
        "host": "localhost",
        "port": "5433"
    }
    try:
        conn = psycopg2.connect(**connection_params)
        #print("Connection successful")
        return conn
    except Exception as e:
        print(f"Failed to connect. An error occurred: {e}")
        return None
    
# close connection to postgres DB
def close_connection(conn):
    try:
        conn.close()
        #print("Connection closed")
    except Exception as e:
        print(f"Failed to close connection. An error occurred: {e}")


# get the size of the full database
def get_database_size(conn):
    cur = conn.cursor()
    try:
        # Query to get the size of the database in MB
        cur.execute("SELECT pg_database_size(current_database()) / (1024 * 1024) AS size_mb")
        database_size_mb = cur.fetchone()[0]

    except Exception as e:
        print(f"An error occurred while getting the database size: {e}")
        database_size_mb = None

    # Close the cursor
    cur.close()

    return database_size_mb


# get the size of a table in the database
def get_table_size(conn, table_name):
    cur = conn.cursor()
    try:
        # Query to get the size of the table in MB
        cur.execute(f"SELECT pg_total_relation_size('{table_name}') / (1024 * 1024) AS size_mb")
        table_size_mb = cur.fetchone()[0]

    except Exception as e:
        print(f"An error occurred while getting the size of table '{table_name}': {e}")
        table_size_mb = None

    # Close the cursor
    cur.close()

    return table_size_mb

# get the size of a table  and its row count 
def get_table_size_and_row_count(conn, table_name):
    query = f"""
            SELECT pg_total_relation_size('{table_name}') / (1024 * 1024) AS size_mb, (SELECT COUNT(*) FROM {table_name}) AS row_count
            """
    try:
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchone()
        table_info =  {"size": result[0], "row_count": result[1]}
        
        # Close the cursor
        cur.close()

    except Exception as e:
        print(f"An error occurred while getting the size and row count of table '{table_name}': {e}")
        table_info = None    
     
    return table_info


# get sizes of all tables in the DB
def get_all_table_sizes(conn):
    try:
        # Create a cursor object
        cur = conn.cursor()

        # Execute the query to get the sizes of all tables
        query = """
        SELECT
            relname AS table_name,
            pg_total_relation_size(relid) / (1024 * 1024) AS total_size_mb
        FROM
            pg_catalog.pg_statio_user_tables
        ORDER BY
            total_size_mb DESC;
        """
        cur.execute(query)
        # Fetch the results
        table_sizes = cur.fetchall()
        # convert to dictionary
        table_sizes = dict(table_sizes)

        # Close the cursor
        cur.close()

    except Exception as e:
        print(f"An error occurred while getting the sizes of all tables: {e}")
        table_sizes = None
    
    return table_sizes


# get all primary clustered indexes and their oid
def get_primary_key_indexes(conn):
    try:
        cur = conn.cursor()

        # SQL query to get primary key indexes and their OIDs
        query = """
        SELECT
            c.relname AS index_name,
            c.oid AS index_oid
        FROM
            pg_index i
        JOIN
            pg_class c ON c.oid = i.indexrelid
        WHERE
            i.indisprimary;
        """

        # Execute the query
        cur.execute(query)

        # Fetch all results
        indexes = cur.fetchall()

        # Close the cursor and connection
        cur.close()
        conn.close()

        # Return the primary key indexes along with their OIDs
        pk_indexes = {}
        for index_name, index_oid in indexes:
            if index_name.startswith("pk_"):
                pk_indexes[index_name] = index_oid

        return pk_indexes

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


# execute a query on postgres DB
def execute_query(conn, query_string, with_explain=True,  return_access_info=False, print_results=False):    
    # Create a cursor object
    cur = conn.cursor()
    try:
        # Execute a query
        if with_explain:
            query_string = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query_string}"
            cur.execute(query_string)
            execution_plan_json = cur.fetchone()[0]
            # Parse the JSON result
            if print_results: print(json.dumps(execution_plan_json, indent=2))
            rows = None
            total_execution_time = execution_plan_json[0]['Plan'].get('Actual Total Time')

            if return_access_info:
                table_access_info, index_access_info, bitmap_heapscan_info = extract_access_info(execution_plan_json[0]['Plan'])
            
        else:        
            cur.execute(query_string)
            # Fetch and print the results (first 10 rows only)
            rows = cur.fetchall()
            if print_results:
                i = 0
                for row in rows:
                    print(row)
                    i += 1
                    if i >= 10: break

    except Exception as e:
        print(f"An error occurred while executing query: {e}")
        rows = None
        total_execution_time = None
        table_access_info = None
        index_access_info = None
        bitmap_heapscan_info = None

    # Close the cursor
    cur.close()
    if return_access_info:
        return total_execution_time, rows, table_access_info, index_access_info, bitmap_heapscan_info

    return total_execution_time, rows


# create a real index
def create_index(conn, index_object):
    table_name = index_object.table_name
    index_id = index_object.index_id
    index_columns = index_object.index_columns
    include_columns = index_object.include_columns

    # Create a cursor object
    cur = conn.cursor()
    try:

        # Construct the index definition string
        index_columns_str = ', '.join(index_columns)
        include_clause = f" INCLUDE ({', '.join(include_columns)})" if include_columns else ""
        index_definition = f"CREATE INDEX {index_id} ON {table_name} ({index_columns_str}){include_clause}"

        start_time = time.perf_counter()
        # Execute the CREATE INDEX statement to create a real index
        cur.execute(index_definition)
        conn.commit()  # Commit the transaction to make the index creation permanent
        #print(f"Real index '{index_id}' created successfully")


        # Get the size of the newly created index (in Mb)
        cur.execute(f"SELECT pg_relation_size('{index_id}') / 1048576.0")
        index_size = cur.fetchone()[0]

        # Retrieve the OID of the newly created index
        cur.execute(f"SELECT oid FROM pg_class WHERE relname = '{index_id.lower()}'")
        index_oid = cur.fetchone()[0]
 
        end_time = time.perf_counter()
        creation_time = (end_time - start_time)*1000
        print(f"Successfully created index: '{index_id}', size: {index_size} MB, creation time: {creation_time:.2f} ms")

    except Exception as e:
        print(f"An error occurred while creating the real index: {e}")
        index_size = None
        index_oid = None
        creation_time = None

    # Close the cursor
    cur.close()

    return index_oid, index_size, creation_time


# bulk create real indexes
def bulk_create_indexes(conn, index_objects):
    for index_object in index_objects:
        index_oid, index_size_mb, creation_time = create_index(conn, index_object)
        index_object.size = float(index_size_mb)
        index_object.creation_time = creation_time
        index_object.current_oid = index_oid


# drop a real index
def drop_index(conn, index_object):
    index_id = index_object.index_id

    # Create a cursor object
    cur = conn.cursor()
    try:
        # Construct the DROP INDEX statement
        drop_index_query = f"DROP INDEX IF EXISTS {index_id}"

        # Execute the DROP INDEX statement
        cur.execute(drop_index_query)
        conn.commit()  # Commit the transaction to make the index drop permanent

        # erase oid
        index_object.current_oid = None

        print(f"Index '{index_id}' dropped successfully")

    except Exception as e:
        print(f"An error occurred while dropping the index: {e}")

    # Close the cursor
    cur.close()


# bulk drop real indexes
def bulk_drop_indexes(conn, index_objects):
    for index_object in index_objects:
        drop_index(conn, index_object)


# drop all existing indexes in the DB
def drop_all_indexes(conn):
    cur = conn.cursor()
    try:
        # Query to find all indexes that are not part of primary keys or unique constraints
        query = """
        SELECT indexname, tablename
        FROM pg_indexes
        WHERE schemaname = 'public'
        AND indexname NOT IN (
            SELECT conname
            FROM pg_constraint
            WHERE contype IN ('p', 'u')
        );
        """
        
        # Execute the query to get all indexes
        cur.execute(query)
        indexes = cur.fetchall()

        # Drop each index
        for indexname, tablename in indexes:
            drop_index_query = f"DROP INDEX IF EXISTS {indexname}"
            cur.execute(drop_index_query)
            print(f"Index '{indexname}' on table '{tablename}' dropped successfully")

        # Commit the transaction
        conn.commit()

    except Exception as e:
        print(f"An error occurred while dropping indexes: {e}")

    # Close the cursor
    cur.close()


# obtain query optimizers cost estimate without executing the query
def get_query_cost_estimate(conn, query, show_plan=False):
    # Create a cursor object
    cur = conn.cursor()
    try:
        # Execute a query
        query = f"EXPLAIN (FORMAT JSON) {query}"
        cur.execute(query)
        # Fetch and print the results
        execution_plan_json = cur.fetchone()[0]  # This is a list containing a single dictionary
        total_cost = execution_plan_json[0]['Plan']['Total Cost']

        # Initialize a stack for iterative traversal
        stack = [execution_plan_json[0]['Plan']]
        scan_costs = {}
        # Iteratively process each node
        while stack:
            plan = stack.pop()
            node_type = plan.get('Node Type', '')
            if 'Scan' in node_type:
                if node_type in scan_costs:
                    scan_costs[node_type] += plan.get('Total Cost')
                else:
                    scan_costs[node_type] = plan.get('Total Cost')
            # Add subplans to the stack for further processing
            for subplan in plan.get('Plans', []):
                stack.append(subplan)

        if show_plan:
            print(json.dumps(execution_plan_json, indent=2))

    except Exception as e:
        print(f"An error occurred while obtaining query optimizer cost estimate: {e}")
        total_cost = None
        scan_costs = None

    # Close the cursor
    cur.close()

    return total_cost, scan_costs



# assign hypothetical index sizes to the given index objects
def get_hypothetical_index_sizes(conn, index_objects):
    # create hypothetical indexes and get their estimated size
    index_size = {}
    for index in index_objects:
        index_oid, index_size_mb = create_hypothetical_index(conn, index, return_size=True)
        index_size[index.index_id] = float(index_size_mb)

    # drop all the hypothetical indexes
    bulk_drop_hypothetical_indexes(conn)   

    return index_size


# create a hypothetical index
def create_hypothetical_index(conn, index_object, return_size=False):
    table_name = index_object.table_name
    index_id = index_object.index_id
    index_columns = index_object.index_columns
    include_columns = index_object.include_columns

    # Create a cursor object
    cur = conn.cursor()
    try:
        # Construct the index definition string
        index_columns_str = ', '.join(index_columns)
        include_clause = f" INCLUDE ({', '.join(include_columns)})" if include_columns else ""
        index_definition = f"CREATE INDEX {index_id} ON {table_name} ({index_columns_str}){include_clause}"

        # Use hypopg to create a hypothetical index
        cur.execute(f"SELECT * FROM hypopg_create_index('{index_definition}')")
        index_oid = cur.fetchone()[0]  # Get the OID of the hypothetical index
        #print(f"Hypothetical index '{index_id}' created successfully")

        if return_size:
            # Get the estimated size of the hypothetical index
            cur.execute(f"SELECT hypopg_relation_size({index_oid})")
            index_size_bytes = cur.fetchone()[0]
            index_size_mb = index_size_bytes / (1024 * 1024)  # Convert bytes to megabytes
            #print(f"Estimated size of hypothetical index '{index_id}': {index_size_mb:.2f} MB")

    except Exception as e:
        print(f"An error occurred while creating the hypothetical index: {e}")
        index_oid = None
        index_size_mb = None

    # Close the cursor
    cur.close()

    if return_size:
        return index_oid, index_size_mb
    else:
        return index_oid

# bulk create hypothetical indexes
def bulk_create_hypothetical_indexes(conn, index_objects, return_size=False):
    hypo_indexes = []
    for index_object in index_objects:
        if return_size:
            index_oid, index_size_mb = create_hypothetical_index(conn, index_object, return_size)
            if index_oid:
                hypo_indexes.append((index_oid, index_size_mb))
        else:
            index_oid = create_hypothetical_index(conn, index_object)
            if index_oid:
                hypo_indexes.append(index_oid)

    return hypo_indexes


# drop the specified hypothetical index
def drop_hypothetical_index(conn, index_oid):
    """Drops a specific hypothetical index by its OID."""
    cur = conn.cursor()
    try:
        # Drop a specific hypothetical index
        cur.execute(f"SELECT * FROM hypopg_drop_index({index_oid})")
        #print(f"Hypothetical index with OID {index_oid} has been dropped successfully.")

    except Exception as e:
        print(f"An error occurred while dropping the hypothetical index with OID {index_oid}: {e}")

    finally:
        cur.close()


# drop all hypothetical indexes
def bulk_drop_hypothetical_indexes(conn):
    """Drops all hypothetical indexes."""
    cur = conn.cursor()
    try:
        # Drop all hypothetical indexes
        cur.execute("SELECT * FROM hypopg_reset()")
        #print("All hypothetical indexes have been dropped successfully.")

    except Exception as e:
        print(f"An error occurred while dropping all hypothetical indexes: {e}")

    finally:
        cur.close()


# show all available hypothetical indexes
def list_hypothetical_indexes(conn):
    """Lists all hypothetical indexes."""
    cur = conn.cursor()
    try:
        # Query to list all hypothetical indexes
        cur.execute("SELECT * FROM hypopg_list_indexes")
        hypothetical_indexes = cur.fetchall()

        # Print the hypothetical indexes
        if hypothetical_indexes:
            print("Hypothetical Indexes:")
            for index in hypothetical_indexes:
                print(f"OID: {index[0]}, Definition: {index[1]}")
        else:
            print("No hypothetical indexes found.")

    except Exception as e:
        print(f"An error occurred while listing hypothetical indexes: {e}")

    finally:
        cur.close()


# get the estimated cost with hypothetical indexes
def get_query_cost_estimate_hypo_indexes(conn, query, show_plan=False):
    """Analyzes a query to obtain the estimated cost and check usage of index scans."""
    cur = conn.cursor()
    try:
        # Execute the EXPLAIN query
        explain_query = f"EXPLAIN (FORMAT JSON) {query}"
        cur.execute(explain_query)
        plan_json = cur.fetchone()[0]  # Fetch the JSON plan

        if show_plan:
            print(json.dumps(plan_json, indent=2))

        # Extract the total cost
        total_cost = plan_json[0]['Plan']['Total Cost']
 
        # Check for index scan usage
        table_access_info, index_access_info, bitmap_heapscan_info = extract_access_info(plan_json[0]['Plan'])
        #print(f"Index Access Info: {index_access_info}")
        indexes_used = []
        if len(index_access_info)>0:
            #print("Indexes used in the query plan:")
            for index_name, index_info in index_access_info.items():
                scan_type = index_info['scan_type']
                scan_cost = index_info['total_cost']
                #print(f"- {index_name}")
                # only consider hypothetical index scans
                if '<' in index_name:
                    index_oid = int(index_name.split('<')[1].split('>')[0])
                    indexes_used.append((index_oid, scan_type, scan_cost, 2)) # 2 for secondary indexes
                else:
                    indexes_used.append((index_name, scan_type, scan_cost, 1)) # 1 for primary clustered indexes      

        else:
            print("No index scans were explicitly noted in the query plan.")

    except Exception as e:
        print(f"An error occurred while analyzing the query: {e}")
        total_cost = None
        indexes_used = None

    finally:
        cur.close()

    return total_cost, indexes_used  


# hypothetical query execution speedup due to a configuration change from X_old to X_new 
def hypo_query_speedup(conn, query_object, X_old, X_new, currently_materialized_indexes=[]):
    # get estimated cost in configuration X_old
    cost_old = hypo_query_cost(conn, query_object, X_old, currently_materialized_indexes)
    # get estimated cost in configuration X_new
    cost_new = hypo_query_cost(conn, query_object, X_new, currently_materialized_indexes)

    # calculate speedup
    if cost_old and cost_new:
        speedup = cost_old / cost_new
    else:
        speedup = None

    return speedup, cost_new


# estimated cost of a query in a given hypothetical configuration X
def hypo_query_cost(conn, query_object, X, currently_materialized_indexes=[]):
    # create hypothetical indexes for X
    X_indexes = bulk_create_hypothetical_indexes(conn, X)
    # hide existing indexes
    bulk_hide_indexes(conn, currently_materialized_indexes)
    # get estimated cost in configuration X
    cost, indexes_used = get_query_cost_estimate_hypo_indexes(conn, query_object.query_string, show_plan=False)
    # drop hypothetical indexes for X
    bulk_drop_hypothetical_indexes(conn)
    # unhide existing indexes
    bulk_unhide_indexes(conn, currently_materialized_indexes)
    return cost


# Hides an existing index using HypoPG.
def hide_index(conn, index_oid):
    cur = conn.cursor()
    try:
        # Hide the index using its OID
        cur.execute(f"SELECT hypopg_hide_index({index_oid})")
        result = cur.fetchone()[0]
        #print(f"Index with OID {index_oid} hidden: {result}")
        return result
    except Exception as e:
        print(f"An error occurred while hiding the index: {e}")
        return False
    finally:
        cur.close()


def bulk_hide_indexes(conn, index_objects):
    for index in index_objects:
        if index.current_oid is not None:
            hide_index(conn, index.current_oid)
        else:
            print(f"Index '{index.index_id}' has no OID (not materialized yet). Skipping...")


# Unhides a previously hidden index using HypoPG.
def unhide_index(conn, index_oid):
    cur = conn.cursor()
    try:
        # Unhide the index using its OID
        cur.execute(f"SELECT hypopg_unhide_index({index_oid})")
        result = cur.fetchone()[0]
        #print(f"Index with OID {index_oid} unhidden: {result}")
        return result
    except Exception as e:
        print(f"An error occurred while unhiding the index: {e}")
        return False
    finally:
        cur.close()


def bulk_unhide_indexes(conn, index_objects):
    for index in index_objects:
        if index.current_oid is not None:
            unhide_index(conn, index.current_oid)        
        else:
            print(f"Index '{index.index_id}' has no OID (not materialized yet). Skipping...")


# find index scans in the query plan
def find_index_scans(plan):
    """Iterative search for index scan operations and extracts index names."""
    indexes = list()  # Use a set to store unique index names
    nodes_to_visit = [plan]  # Initialize the stack with the root plan node

    while nodes_to_visit:
        current_node = nodes_to_visit.pop()

        # Check for nodes that indicate index scans
        if current_node.get('Node Type') in ['Index Scan', 'Index Only Scan', 'Bitmap Index Scan']:
            index_name = current_node.get('Index Name')
            if index_name:
                indexes.append((index_name, current_node.get('Node Type'), current_node.get('Total Cost')))

        # Add subplans to the stack
        nodes_to_visit.extend(current_node.get('Plans', []))

    return indexes


# extract information about all access methods used in the query plan
def extract_access_info(plan):
    """Iterative search for all access methods used in the query plan."""
    # dictionaries to store access method information
    table_access_info = {}  
    index_access_info = {}
    bitmap_heapscan_info = {}
    nodes_to_visit = [plan]  # Initialize the stack with the root plan node

    while nodes_to_visit:
        current_node = nodes_to_visit.pop()

        # check for nodes that indicate index scans
        if current_node.get('Node Type') in ['Index Scan', 'Index Only Scan', 'Bitmap Index Scan']:
            index_name = current_node.get('Index Name')
            if index_name:
                scan_type = current_node.get('Node Type')
                total_cost = current_node.get('Total Cost')
                actual_rows = current_node.get('Actual Rows')
                actual_startup_time = current_node.get('Actual Startup Time')
                actual_total_time = current_node.get('Actual Total Time')  
                table_name = current_node.get('Relation Name')      
                shared_hit_blocks = current_node.get('Shared Hit Blocks')
                shared_read_blocks = current_node.get('Shared Read Blocks')
                local_hit_blocks = current_node.get('Local Hit Blocks')
                local_read_blocks = current_node.get('Local Read Blocks')
                index_access_info[index_name] = {'table':table_name, 'scan_type': scan_type, 'total_cost': total_cost, 'actual_rows': actual_rows, 'actual_startup_time': actual_startup_time, 'actual_total_time': actual_total_time, 'shared_hit_blocks': shared_hit_blocks, 'shared_read_blocks': shared_read_blocks, 'local_hit_blocks': local_hit_blocks, 'local_read_blocks': local_read_blocks}  

        elif current_node.get('Node Type') in ['Bitmap Heap Scan']:
            table_name = current_node.get('Relation Name')
            if table_name:
                total_cost = current_node.get('Total Cost')
                scan_type = current_node.get('Node Type')
                actual_rows = current_node.get('Actual Rows')
                actual_startup_time = current_node.get('Actual Startup Time')
                actual_total_time = current_node.get('Actual Total Time')  
                table_name = current_node.get('Relation Name')      
                shared_hit_blocks = current_node.get('Shared Hit Blocks')
                shared_read_blocks = current_node.get('Shared Read Blocks')
                local_hit_blocks = current_node.get('Local Hit Blocks')
                local_read_blocks = current_node.get('Local Read Blocks')
                bitmap_heapscan_info[table_name] = {'table':table_name, 'scan_type': scan_type, 'total_cost': total_cost, 'actual_rows': actual_rows, 'actual_startup_time': actual_startup_time, 'actual_total_time': actual_total_time, 'shared_hit_blocks': shared_hit_blocks, 'shared_read_blocks': shared_read_blocks, 'local_hit_blocks': local_hit_blocks, 'local_read_blocks': local_read_blocks}


        # check for nodes that indicate table sequential scans
        elif current_node.get('Node Type') in ['Seq Scan']:
            table_name = current_node.get('Relation Name')
            if table_name:
                total_cost = current_node.get('Total Cost')
                actual_rows = current_node.get('Actual Rows')
                actual_startup_time = current_node.get('Actual Startup Time')
                actual_total_time = current_node.get('Actual Total Time')
                shared_hit_blocks = current_node.get('Shared Hit Blocks')
                shared_read_blocks = current_node.get('Shared Read Blocks')
                local_hit_blocks = current_node.get('Local Hit Blocks')
                local_read_blocks = current_node.get('Local Read Blocks')
                table_access_info[table_name] = {'total_cost': total_cost, 'actual_rows': actual_rows, 'actual_startup_time': actual_startup_time, 'actual_total_time': actual_total_time, 'shared_hit_blocks': shared_hit_blocks, 'shared_read_blocks': shared_read_blocks, 'local_hit_blocks': local_hit_blocks, 'local_read_blocks': local_read_blocks}


        # Add subplans to the stack
        nodes_to_visit.extend(current_node.get('Plans', []))

    return table_access_info, index_access_info, bitmap_heapscan_info



# measure sequential scan time for a table (make sure there are no secondary indexes on the table)
def get_sequential_scan_time(conn, table_name):
    cur = conn.cursor()
    try:
        # Construct a query that will likely result in a sequential scan
        query_string = f"SELECT * FROM {table_name};"
        # get the execution plan
        explain_query = f"EXPLAIN (ANALYZE, FORMAT JSON) {query_string}"
        cur.execute(explain_query)
        execution_plan_json = cur.fetchone()[0]

        #total_execution_time = execution_plan_json[0]['Plan'].get('Actual Total Time')

        # Extract the time taken by the sequential scan operator
        """    
        seq_scan_time = None
        seq_scan_found = False
        for plan_node in execution_plan_json[0]['Plan']:
            if plan_node['Node Type'] == 'Seq Scan':
                # Extract the actual time from the plan node
                seq_scan_time = plan_node['Actual Total Time']
                seq_scan_found = True
                break
        """
        nodes_to_visit = [execution_plan_json[0]['Plan']]  # Initialize the stack with the root plan node
        while nodes_to_visit:
            current_node = nodes_to_visit.pop()

            # Check for nodes that indicate index scans
            if current_node.get('Node Type') == 'Seq Scan':
                seq_scan_time = current_node.get('Actual Total Time')
                seq_scan_found = True
                break

            # Add subplans to the stack
            nodes_to_visit.extend(current_node.get('Plans', []))

        # Close the cursor
        cur.close()       

        if not seq_scan_found:
            print("Sequential scan operator not found in the query plan.")
        else:
            print(f"Sequential scan time for table '{table_name}': {seq_scan_time:.2f} ms")    
        

    except Exception as e:
        print(f"An error occurred: {e}")
        seq_scan_time = None

    return seq_scan_time    


class Index:
    def __init__(self, table_name, index_id, index_columns, include_columns=()):
        self.table_name = table_name
        self.index_id = index_id
        self.index_columns = index_columns
        self.include_columns = include_columns
        self.size = None
        self.creation_time = None
        self.current_oid = None

    def __str__(self):
        return f"Index name: {self.index_id}, Key cols: {self.index_columns}, Include cols: {self.include_columns}, Current OID: {self.current_oid}"


# assign a unique id to a given index
def get_index_id(index_cols, table_name, include_cols=()):
    if include_cols:
        include_col_names = '_'.join(tuple(map(lambda x: x[0:4], include_cols))).lower()
        index_id = 'ixn_' + table_name.lower() + '_' + '_'.join(index_cols).lower() + '_' + include_col_names
    else:
        index_id = 'ix_' + table_name.lower() + '_' + '_'.join(index_cols).lower()
    
    # truncate to 128 chars
    index_id = index_id[:127]
    return index_id


# enumerate candidate indexes that could benefit a given query
# TODO: option for max number of index key/include columns 
# TODO: option for including group_by/order_by columns (since some queries may benefit from having sorted data inside indexes and avoid sorting)
def extract_query_indexes(query_object, max_key_columns=None, include_cols=False):
    # use only the predicates and payloads for now
    predicates = query_object.predicates
    payload = query_object.payload

    candidate_indexes = {}
    for table in predicates:
        predicate_cols = predicates[table]
        payload_cols = payload[table] if table in payload else []
        # create permutations of the predicate columns
        if max_key_columns:
            max_predicates = min(max_key_columns, len(predicate_cols))
        else:
            max_predicates = len(predicate_cols)    
        for i in range(1, max_predicates+1):
            for index_key_cols in itertools.permutations(predicate_cols, i):
                index_id = get_index_id(index_key_cols, table)
                if index_id not in candidate_indexes:
                    candidate_indexes[index_id] = Index(table, index_id, index_key_cols)
                if include_cols:
                    # remove payload columns that intersect with key columns
                    payload_columns_filtered = tuple(set(payload_cols) - set(index_key_cols))
                    for j in range(1, len(payload_columns_filtered)+1):
                        for index_include_cols in itertools.combinations(payload_columns_filtered, j):
                            # remove include columns that intersect with key columns
                            if index_include_cols:
                                index_id = get_index_id(index_key_cols, table, index_include_cols)
                                if index_id not in candidate_indexes:
                                    candidate_indexes[index_id] = Index(table, index_id, index_key_cols, index_include_cols)  

    candidate_indexes = list(candidate_indexes.values())

    return candidate_indexes            


# create index objects for primary clustered indexes
def ssb_pk_index_objects():
    index_cols = {'lineorder': ['lo_orderkey', 'lo_linenumber'], 'customer': ['c_custkey'], 'supplier': ['s_suppkey'], 'part': ['p_partkey'], 'dwdate': ['d_datekey']}
    conn = create_connection()
    pk_index_oids = get_primary_key_indexes(conn)
    close_connection(conn)
    pk_indexes = []
    for table_name, index_columns in index_cols.items():
        index_id = f"pk_{table_name}"
        index_object = Index(table_name, index_id, index_columns)
        index_object.current_oid = pk_index_oids.get(index_id)
        pk_indexes.append(index_object)

    return pk_indexes



























