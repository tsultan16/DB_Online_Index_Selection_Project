"""
    A collection of Util Functions for talking to PostgreSQL database
"""

import psycopg2
import json
import itertools
import time


# create connection to postgres DB
def create_connection():
    # Define your connection parameters
    connection_params = {
        "dbname": "SSB10",
        "user": "postgres",
        "password": "greatpond501",
        "host": "localhost",
        "port": "5432"
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

# execute a query on postgres DB
def execute_query(conn, query, with_explain=True, print_results=False):
    # Create a cursor object
    cur = conn.cursor()
    try:
        # Execute a query
        if with_explain:
            query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
            cur.execute(query)
            execution_plan_json = cur.fetchone()[0]
            # Parse the JSON result
            if print_results: print(json.dumps(execution_plan_json, indent=2))
            rows = None
            total_execution_time = execution_plan_json[0]['Plan'].get('Actual Total Time')

        else:        
            cur.execute(query)
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

    # Close the cursor
    cur.close()
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

        end_time = time.perf_counter()
        creation_time = end_time - start_time

        # Get the size of the newly created index
        cur.execute(f"SELECT pg_size_pretty(pg_relation_size('{index_id}'))")
        index_size = cur.fetchone()[0]
 
        print(f"Successfully created index '{index_id}': {index_size}, creation time: {creation_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred while creating the real index: {e}")
        index_size = None
        creation_time = None

    # Close the cursor
    cur.close()

    return index_size, creation_time


# bulk create real indexes
def bulk_create_indexes(conn, index_objects):
    for index_object in index_objects:
        index_size_mb, creation_time = create_index(conn, index_object)
        index_object.size = index_size_mb
        index_object.creation_time = creation_time


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
        index_scans = find_index_scans(plan_json[0]['Plan'])
        indexes_used = []

        if len(index_scans)>0:
            #print("Indexes used in the query plan:")
            for index_name, scan_type, scan_cost in index_scans:
                #print(f"- {index_name}")
                # only consider hypothetical index scans
                if '<' in index_name:
                    index_oid = int(index_name.split('<')[1].split('>')[0])
                    indexes_used.append((index_oid, scan_type, scan_cost)) 
                

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
def hypo_query_speedup(conn, query_object, X_old, X_new):
    # create hypothetical indexes for X_old 
    X_old_indexes = bulk_create_hypothetical_indexes(conn, X_old)
    # get estimated cost in configuration X_old
    cost_old, indexes_used = get_query_cost_estimate_hypo_indexes(conn, query_object.query_string, show_plan=False)
    # drop hypothetical indexes for X_old
    bulk_drop_hypothetical_indexes(conn)

    # create hypothetical indexes for X_new
    X_new_indexes = bulk_create_hypothetical_indexes(conn, X_new)
    # get estimated cost in configuration X_new
    cost_new, indexes_used = get_query_cost_estimate_hypo_indexes(conn, query_object.query_string, show_plan=False)
    # drop hypothetical indexes for X_new
    bulk_drop_hypothetical_indexes(conn)

    # calculate speedup
    if cost_old and cost_new:
        speedup = cost_old / cost_new
    else:
        speedup = None

    return speedup, cost_new



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


class Index:
    def __init__(self, table_name, index_id, index_columns, include_columns=()):
        self.table_name = table_name
        self.index_id = index_id
        self.index_columns = index_columns
        self.include_columns = include_columns
        self.size = None
        self.creation_time = None

    def __str__(self):
        return f"Index name: {self.index_id}, Key cols: {self.index_columns}, Include cols: {self.include_columns}"


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


# enumerate candidate indexes that could benefit a given query
# TODO: option for max number of index key/include columns 
def extract_query_indexes(query_object, max_key_columns=None, include_cols=False):
    # use only the predicates and payloads for now
    predicates = query_object.predicates
    payload = query_object.payload
    #order_by = query_object.order_by
    #group_by = query_object.group_by

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




























