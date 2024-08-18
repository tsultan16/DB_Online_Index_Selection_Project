"""
    A collection of Util Functions for talking to PostgreSQL database
"""

import psycopg2
import json
import itertools



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
        print("Connection closed")
    except Exception as e:
        print(f"Failed to close connection. An error occurred: {e}")


# execute a query on postgres DB
def execute_query(conn, query, with_explain=False, print_results=True):
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

        else:        
            cur.execute(query)
            # Fetch and print the results
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

    # Close the cursor
    cur.close()
    return rows



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
        cost_estimate = execution_plan_json[0]['Plan']['Total Cost']

        if show_plan:
            print(json.dumps(execution_plan_json, indent=2))

    except Exception as e:
        print(f"An error occurred while obtaining query optimizer cost estimate: {e}")
        cost_estimate = None

    # Close the cursor
    cur.close()

    return cost_estimate


# create a hypothetical index
def create_hypothetical_index(conn, index_object):
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
        print(f"Hypothetical index '{index_id}' created successfully")

        # Get the estimated size of the hypothetical index
        cur.execute(f"SELECT hypopg_relation_size({index_oid})")
        index_size_bytes = cur.fetchone()[0]
        index_size_mb = index_size_bytes / (1024 * 1024)  # Convert bytes to megabytes
        print(f"Estimated size of hypothetical index '{index_id}': {index_size_mb:.2f} MB")

    except Exception as e:
        print(f"An error occurred while creating the hypothetical index: {e}")
        index_oid = None
        index_size_mb = None

    # Close the cursor
    cur.close()

    return index_oid, index_size_mb


# drop the specified hypothetical index
def drop_hypothetical_index(conn, index_oid):
    """Drops a specific hypothetical index by its OID."""
    cur = conn.cursor()
    try:
        # Drop a specific hypothetical index
        cur.execute(f"SELECT * FROM hypopg_drop_index({index_oid})")
        print(f"Hypothetical index with OID {index_oid} has been dropped successfully.")

    except Exception as e:
        print(f"An error occurred while dropping the hypothetical index with OID {index_oid}: {e}")

    finally:
        cur.close()


# drop all hypothetical indexes
def drop_all_hypothetical_indexes(conn):
    """Drops all hypothetical indexes."""
    cur = conn.cursor()
    try:
        # Drop all hypothetical indexes
        cur.execute("SELECT * FROM hypopg_reset()")
        print("All hypothetical indexes have been dropped successfully.")

    except Exception as e:
        print(f"An error occurred while dropping all hypothetical indexes: {e}")

    finally:
        cur.close()



class Index:
    def __init__(self, table_name, index_id, index_columns, include_columns=()):
        self.table_name = table_name
        self.index_id = index_id
        self.index_columns = index_columns
        self.include_columns = include_columns

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
def extract_query_indexes(query_object):
    # use only the predicates for now
    predicates = query_object.predicates
    #payload = query_object.payload
    #order_by = query_object.order_by
    #group_by = query_object.group_by

    candidate_indexes = []
    for table in predicates:
        predicate_cols = predicates[table]
        # create permutations of the predicate columns
        for i in range(1, len(predicate_cols)+1):
            for index_key_cols in itertools.combinations(predicate_cols, i):
                candidate_indexes.append(Index(table, get_index_id(index_key_cols, table), index_key_cols))

    return candidate_indexes            



























