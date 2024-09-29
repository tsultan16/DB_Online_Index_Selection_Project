"""
    TPC-H query predicates parser
"""

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re


def parse_tpch_query_1(query):
    # Extract the base date from the query
    base_date_str = "1998-12-01"
    base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
    
    # Use a regular expression to find the number of days in the INTERVAL
    match = re.search(r"INTERVAL '(\d+) days'", query)
    if match:
        days = int(match.group(1))
    else:
        raise ValueError("Could not find the number of days in the query.")
    
    # Calculate the exact date by subtracting the extracted number of days
    exact_date = base_date - timedelta(days=days)
    exact_date_str = exact_date.strftime("%Y-%m-%d")
    
    # Construct the predicate dictionary
    predicate_dict = {
        "lineitem": [
            {"column": "l_shipdate", "operator": "<=", "value": f"'{exact_date_str}'", "join": False}
        ]    
    }
    
    return predicate_dict

def parse_tpch_query_2(query):
    # Use string operations to extract predicate values
    # Extract p_size
    p_size_start = query.find("p_size =") + len("p_size =")
    p_size_end = query.find("\n", p_size_start)
    p_size_value = query[p_size_start:p_size_end].strip()

    # Extract p_type without regex
    p_type_start = query.find("p_type LIKE '%' || '") + len("p_type LIKE '%' || '")
    p_type_end = query.find("'", p_type_start)
    p_type_value = query[p_type_start:p_type_end].strip()

    # Extract r_name
    r_name_start = query.find("r_name = '") + len("r_name = '")
    r_name_end = query.find("'", r_name_start)
    r_name_value = query[r_name_start:r_name_end].strip()

    # Construct the predicate dictionary
    predicate_dict = {
        "part": [
            {"column": "p_size", "operator": "=", "value": p_size_value, "join": False},
            {"column": "p_type", "operator": "LIKE", "value": f"'%{p_type_value}'", "join": False},
            {"column": "p_partkey", "operator": "=", "value": "ps_partkey", "join": True}  
        ],
        "region": [
            {"column": "r_name", "operator": "=", "value": r_name_value, "join": False}
        ],
        "supplier": [
            {"column": "s_suppkey", "operator": "=", "value": "ps_suppkey", "join": True},
            {"column": "s_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ], 
        "nation": [
            {"column": "n_nationkey", "operator": "=", "value": "s_nationkey", "join": True}
        ]
    }
    
    return predicate_dict


def parse_tpch_query_3(query):
    # Use regular expressions to extract predicate values
    c_mktsegment_match = re.search(r"c_mktsegment\s*=\s*'([^']+)'", query)
    o_orderdate_match = re.search(r"o_orderdate\s*<\s*CAST\('([^']+)' AS date\)", query)
    l_shipdate_match = re.search(r"l_shipdate\s*>\s*CAST\('([^']+)' AS date\)", query)

    # Extracted values
    c_mktsegment_value = c_mktsegment_match.group(1) if c_mktsegment_match else None
    o_orderdate_value = o_orderdate_match.group(1) if o_orderdate_match else None
    l_shipdate_value = l_shipdate_match.group(1) if l_shipdate_match else None

    # Construct the predicate dictionary
    predicate_dict = {
        "customer": [
            {"column": "c_mktsegment", "operator": "=", "value": f"'{c_mktsegment_value}'", "join": False},
            {"column": "c_custkey", "operator": "=", "value": "o_custkey", "join": True}
        ],
        "orders": [
            {"column": "o_orderkey", "operator": "=", "value": "l_orderkey", "join": True},
            {"column": "o_orderdate", "operator": "<", "value": o_orderdate_value, "join": False}
        ],
        "lineitem": [
            {"column": "l_shipdate", "operator": ">", "value": l_shipdate_value, "join": False}
        ]
    }
    
    return predicate_dict

def parse_tpch_query_4(query):
    # Use string operations to extract predicate values
    # Extract start date
    start_date_str = query.split("o_orderdate >= CAST('")[1].split("' AS date)")[0]
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    # Calculate end date by adding 3 months
    end_date = start_date + timedelta(days=90)  # Approximation for 3 months

    # Format end date as string
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Construct the predicate dictionary
    predicate_dict = {
        "orders": [
            {"column": "o_orderdate", "operator": "range", "value": (start_date_str, end_date_str), "join": False},
        ],
        "lineitem": [
            {"column": "l_orderkey", "operator": "=", "value": "o_orderkey", "join": True},
            {"column": "l_commitdate", "operator": "<", "value": "l_receiptdate", "join": False}
        ]
    }

    return predicate_dict

def parse_tpch_query_5(query):
    # Use regular expressions to extract predicate values
    r_name_match = re.search(r"r_name\s*=\s*'([^']+)'", query)
    o_orderdate_start_match = re.search(r"o_orderdate\s*>=\s*CAST\('([^']+)' AS date\)", query)

    # Extracted values
    r_name_value = r_name_match.group(1) if r_name_match else None
    o_orderdate_start_value = o_orderdate_start_match.group(1) if o_orderdate_start_match else None

    # Calculate the end date by adding 1 year
    if o_orderdate_start_value:
        start_date = datetime.strptime(o_orderdate_start_value, '%Y-%m-%d')
        o_orderdate_end_value = (start_date + relativedelta(years=1)).strftime('%Y-%m-%d')
    else:
        o_orderdate_end_value = None

    # Construct the predicate dictionary
    predicate_dict = {
        "region": [
            {"column": "r_name", "operator": "=", "value": f"'{r_name_value}'", "join": False}
        ],
        "orders": [
            {"column": "o_orderdate", "operator": "range", "value": (o_orderdate_start_value, o_orderdate_end_value), "join": False}
        ],
        "customer": [
            {"column": "c_custkey", "operator": "=", "value": "o_custkey", "join": True},
            {"column": "c_nationkey", "operator": "=", "value": "s_nationkey", "join": True}
        ],
        "lineitem": [
            {"column": "l_orderkey", "operator": "=", "value": "o_orderkey", "join": True},
            {"column": "l_suppkey", "operator": "=", "value": "s_suppkey", "join": True}
        ],
        "supplier": [
            {"column": "s_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ],
        "nation": [
            {"column": "n_regionkey", "operator": "=", "value": "r_regionkey", "join": True}
        ]
    }

    return predicate_dict

def parse_tpch_query_6(query):
    # Use regular expressions to extract predicate values
    l_shipdate_start_match = re.search(r"l_shipdate\s*>=\s*CAST\('([^']+)' AS date\)", query)
    l_discount_match = re.search(r"l_discount\s*between\s*([\d.]+)\s*-\s*0\.01\s*and\s*([\d.]+)\s*\+\s*0\.01", query)
    l_quantity_match = re.search(r"l_quantity\s*<\s*(\d+)", query)

    # Extracted values
    l_shipdate_start_value = l_shipdate_start_match.group(1) if l_shipdate_start_match else None

    # Calculate the end date by adding 1 year
    if l_shipdate_start_value:
        start_date = datetime.strptime(l_shipdate_start_value, '%Y-%m-%d')
        l_shipdate_end_value = (start_date + relativedelta(years=1)).strftime('%Y-%m-%d')
    else:
        l_shipdate_end_value = None

    l_discount_low_value = float(l_discount_match.group(1)) - 0.01 if l_discount_match else None
    l_discount_high_value = float(l_discount_match.group(2)) + 0.01 if l_discount_match else None
    l_quantity_value = l_quantity_match.group(1) if l_quantity_match else None

    # Construct the predicate dictionary
    predicate_dict = {
        "lineitem": [
            {"column": "l_shipdate", "operator": "range", "value": (l_shipdate_start_value, l_shipdate_end_value), "join": False},
            {"column": "l_discount", "operator": "range", "value": (l_discount_low_value, l_discount_high_value), "join": False},
            {"column": "l_quantity", "operator": "<", "value": l_quantity_value, "join": False}
        ]
    }

    return predicate_dict


def parse_tpch_query_7(query):
    # Use regular expressions to extract predicate values
    nation1_match = re.search(r"n1\.n_name\s*=\s*'([^']+)'", query)
    nation2_match = re.search(r"n2\.n_name\s*=\s*'([^']+)'", query)
    l_shipdate_start_match = re.search(r"l_shipdate\s*between\s*CAST\('([^']+)' AS date\)", query)
    l_shipdate_end_match = re.search(r"and\s*CAST\('([^']+)' AS date\)", query)

    # Extracted values
    nation1_value = nation1_match.group(1) if nation1_match else None
    nation2_value = nation2_match.group(1) if nation2_match else None
    l_shipdate_start_value = l_shipdate_start_match.group(1) if l_shipdate_start_match else None
    l_shipdate_end_value = l_shipdate_end_match.group(1) if l_shipdate_end_match else None

    # Construct the predicate dictionary
    predicate_dict = {
        "nation": [
            {"column": "n_name", "operator": "=", "value": f"'{nation1_value}'", "join": False},
            {"column": "n_name", "operator": "=", "value": f"'{nation2_value}'", "join": False}
        ],
        "lineitem": [
            {"column": "l_shipdate", "operator": "range", "value": (l_shipdate_start_value, l_shipdate_end_value), "join": False}
        ],
        "supplier": [
            {"column": "s_suppkey", "operator": "=", "value": "l_suppkey", "join": True},
            {"column": "s_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ],
        "orders": [
            {"column": "o_orderkey", "operator": "=", "value": "l_orderkey", "join": True}
        ],
        "customer": [
            {"column": "c_custkey", "operator": "=", "value": "o_custkey", "join": True},
            {"column": "c_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ]
    }

    return predicate_dict


def parse_tpch_query_8(query):
    # Use regular expressions to extract predicate values
    r_name_match = re.search(r"r_name\s*=\s*'([^']+)'", query)
    o_orderdate_start_match = re.search(r"o_orderdate\s*between\s*CAST\('([^']+)' AS date\)", query)
    o_orderdate_end_match = re.search(r"and\s*CAST\('([^']+)' AS date\)", query)
    p_type_match = re.search(r"p_type\s*=\s*'([^']+)'", query)
    nation_match = re.search(r"case\s+when\s+nation\s*=\s*'([^']+)'", query)

    # Extracted values
    r_name_value = r_name_match.group(1) if r_name_match else None
    o_orderdate_start_value = o_orderdate_start_match.group(1) if o_orderdate_start_match else None
    o_orderdate_end_value = o_orderdate_end_match.group(1) if o_orderdate_end_match else None
    p_type_value = p_type_match.group(1) if p_type_match else None
    nation_value = nation_match.group(1) if nation_match else None

    # Construct the predicate dictionary
    predicate_dict = {
        "region": [
            {"column": "r_name", "operator": "=", "value": f"'{r_name_value}'", "join": False}
        ],
        "orders": [
            {"column": "o_orderdate", "operator": "range", "value": (o_orderdate_start_value, o_orderdate_end_value), "join": False}
        ],
        "part": [
            {"column": "p_type", "operator": "=", "value": f"'{p_type_value}'", "join": False}
        ],
        "supplier": [
            {"column": "s_suppkey", "operator": "=", "value": "l_suppkey", "join": True},
            {"column": "s_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ],
        "lineitem": [
            {"column": "l_partkey", "operator": "=", "value": "p_partkey", "join": True},
            {"column": "l_orderkey", "operator": "=", "value": "o_orderkey", "join": True}
        ],
        "customer": [
            {"column": "o_custkey", "operator": "=", "value": "c_custkey", "join": True},
            {"column": "c_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ],
        "nation": [
            {"column": "n_regionkey", "operator": "=", "value": "r_regionkey", "join": True}
        ],
        "case": [
            {"column": "nation", "operator": "=", "value": f"'{nation_value}'", "join": False}
        ]
    }

    return predicate_dict

def parse_tpch_query_9(query):
    # Use regular expressions to extract predicate values
    p_name_match = re.search(r"p_name\s+like\s+'([^']+)'", query)

    # Extracted values
    p_name_value = p_name_match.group(1) if p_name_match else None

    # Construct the predicate dictionary
    predicate_dict = {
        "part": [
            {"column": "p_name", "operator": "like", "value": f"'{p_name_value}'", "join": False}
        ],
        "supplier": [
            {"column": "s_suppkey", "operator": "=", "value": "l_suppkey", "join": True},
            {"column": "s_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ],
        "lineitem": [
            {"column": "l_suppkey", "operator": "=", "value": "s_suppkey", "join": True},
            {"column": "l_partkey", "operator": "=", "value": "p_partkey", "join": True},
            {"column": "l_orderkey", "operator": "=", "value": "o_orderkey", "join": True}
        ],
        "partsupp": [
            {"column": "ps_suppkey", "operator": "=", "value": "l_suppkey", "join": True},
            {"column": "ps_partkey", "operator": "=", "value": "l_partkey", "join": True}
        ],
        "orders": [
            {"column": "o_orderkey", "operator": "=", "value": "l_orderkey", "join": True}
        ],
        "nation": [
            {"column": "s_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ]
    }

    return predicate_dict

def parse_tpch_query_10(query):
    # Use regular expressions to extract predicate values
    o_orderdate_start_match = re.search(r"o_orderdate\s*>=\s*CAST\('([^']+)' AS date\)", query)
    l_returnflag_match = re.search(r"l_returnflag\s*=\s*'([^']+)'", query)

    # Extracted values
    o_orderdate_start_value = o_orderdate_start_match.group(1) if o_orderdate_start_match else None

    # Calculate the end date by adding 3 months
    if o_orderdate_start_value:
        start_date = datetime.strptime(o_orderdate_start_value, '%Y-%m-%d')
        o_orderdate_end_value = (start_date + relativedelta(months=3)).strftime('%Y-%m-%d')
    else:
        o_orderdate_end_value = None

    l_returnflag_value = l_returnflag_match.group(1) if l_returnflag_match else None

    # Construct the predicate dictionary
    predicate_dict = {
        "orders": [
            {"column": "o_orderdate", "operator": "range", "value": (o_orderdate_start_value, o_orderdate_end_value), "join": False}
        ],
        "lineitem": [
            {"column": "l_returnflag", "operator": "=", "value": f"'{l_returnflag_value}'", "join": False},
            {"column": "l_orderkey", "operator": "=", "value": "o_orderkey", "join": True}
        ],
        "customer": [
            {"column": "c_custkey", "operator": "=", "value": "o_custkey", "join": True},
            {"column": "c_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ],
        "nation": [
            {"column": "c_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ]
    }

    return predicate_dict


def parse_tpch_query_11(query):
    # Use regular expressions to extract predicate values
    n_name_match = re.search(r"n_name\s*=\s*'([^']+)'", query)

    # Extracted values
    n_name_value = n_name_match.group(1) if n_name_match else None

    # Construct the predicate dictionary
    predicate_dict = {
        "nation": [
            {"column": "n_name", "operator": "=", "value": f"'{n_name_value}'", "join": False}
        ],
        "partsupp": [
            {"column": "ps_suppkey", "operator": "=", "value": "s_suppkey", "join": True}
        ],
        "supplier": [
            {"column": "s_nationkey", "operator": "=", "value": "n_nationkey", "join": True}
        ],
    }

    return predicate_dict


def parse_tpch_query_12(query):
    # Use regular expressions to extract predicate values
    l_shipmode_match = re.search(r"l_shipmode\s+in\s*\(([^)]+)\)", query)
    l_receiptdate_start_match = re.search(r"l_receiptdate\s*>=\s*CAST\('([^']+)' AS date\)", query)

    # Extracted values
    l_shipmode_value = l_shipmode_match.group(1).replace("'", "").split(", ") if l_shipmode_match else None
    l_receiptdate_start_value = l_receiptdate_start_match.group(1) if l_receiptdate_start_match else None

    # Calculate the end date by adding 1 year
    if l_receiptdate_start_value:
        start_date = datetime.strptime(l_receiptdate_start_value, '%Y-%m-%d')
        l_receiptdate_end_value = (start_date + relativedelta(years=1)).strftime('%Y-%m-%d')
    else:
        l_receiptdate_end_value = None

    # Construct the predicate dictionary
    predicate_dict = {
        "lineitem": [
            {"column": "l_shipmode", "operator": "or", "value": l_shipmode_value, "join": False},
            {"column": "l_commitdate", "operator": "<", "value": "l_receiptdate", "join": False},
            {"column": "l_shipdate", "operator": "<", "value": "l_commitdate", "join": False},
            {"column": "l_receiptdate", "operator": "range", "value": (l_receiptdate_start_value, l_receiptdate_end_value), "join": False}
        ],
        "orders": [
            {"column": "o_orderkey", "operator": "=", "value": "l_orderkey", "join": True}
        ],
    }

    return predicate_dict


def parse_tpch_query_13(query):
    # Use regular expressions to extract predicate values
    join_condition_match = re.search(r"o_comment\s*not\s+like\s+'([^']+)'", query)

    # Extracted values
    join_condition_value = join_condition_match.group(1) if join_condition_match else None

    # Construct the predicate dictionary
    predicate_dict = {
        "customer": [
            {"column": "c_custkey", "operator": "=", "value": "o_custkey", "join": True},
            {"column": "o_comment", "operator": "NOT LIKE", "value": f"'{join_condition_value}'", "join": False}
        ]
    }

    return predicate_dict


def parse_tpch_query_14(query):
    # Use regular expressions to extract predicate values
    p_type_match = re.search(r"p_type\s+like\s+'([^']+)'", query)
    l_shipdate_start_match = re.search(r"l_shipdate\s*>=\s*CAST\('([^']+)' AS date\)", query)

    # Extracted values
    p_type_value = p_type_match.group(1) if p_type_match else None
    l_shipdate_start_value = l_shipdate_start_match.group(1) if l_shipdate_start_match else None

    # Calculate the end date by adding 1 month
    if l_shipdate_start_value:
        start_date = datetime.strptime(l_shipdate_start_value, '%Y-%m-%d')
        l_shipdate_end_value = (start_date + relativedelta(months=1)).strftime('%Y-%m-%d')
    else:
        l_shipdate_end_value = None

    # Construct the predicate dictionary
    predicate_dict = {
        "part": [
            {"column": "p_type", "operator": "LIKE", "value": f"'{p_type_value}'", "join": False}
        ],
        "lineitem": [
            {"column": "l_partkey", "operator": "=", "value": "p_partkey", "join": True},
            {"column": "l_shipdate", "operator": "range", "value": (l_shipdate_start_value, l_shipdate_end_value), "join": False}
        ]
    }

    return predicate_dict



def parse_tpch_query_15(query):
    # Use regular expressions to extract predicate values
    l_shipdate_start_match = re.search(r"l_shipdate\s*>=\s*CAST\('([^']+)' AS date\)", query)

    # Extracted values
    l_shipdate_start_value = l_shipdate_start_match.group(1) if l_shipdate_start_match else None

    # Calculate the end date by adding 3 months
    if l_shipdate_start_value:
        start_date = datetime.strptime(l_shipdate_start_value, '%Y-%m-%d')
        l_shipdate_end_value = (start_date + relativedelta(months=3)).strftime('%Y-%m-%d')
    else:
        l_shipdate_end_value = None

    # Construct the predicate dictionary
    predicate_dict = {
        "lineitem": [
            {"column": "l_shipdate", "operator": "range", "value": (l_shipdate_start_value, l_shipdate_end_value), "join": False}
        ],
        "supplier": [
            {"column": "s_suppkey", "operator": "=", "value": "supplier_no", "join": True}
        ]
    }

    return predicate_dict


def parse_tpch_query_16(query):
    # Use regular expressions to extract predicate values
    p_brand_match = re.search(r"p_brand\s*=\s*'([^']+)'", query)
    p_container_match = re.search(r"p_container\s*=\s*'([^']+)'", query)
    subquery_condition_match = re.search(r"0.2\s*\*\s*avg\(l_quantity\)", query)

    # Extracted values
    p_brand_value = p_brand_match.group(1) if p_brand_match else None
    p_container_value = p_container_match.group(1) if p_container_match else None
    subquery_condition_value = "0.2 * avg(l_quantity)" if subquery_condition_match else None

    # Construct the predicate dictionary
    predicate_dict = {
        "part": [
            {"column": "p_brand", "operator": "=", "value": f"'{p_brand_value}'", "join": False},
            {"column": "p_container", "operator": "=", "value": f"'{p_container_value}'", "join": False},
            {"column": "p_partkey", "operator": "=", "value": "l_partkey", "join": True},
        ],
    }

    return predicate_dict


def parse_tpch_query_17(query):
    # Use regular expressions to extract predicate values
    p_brand_match = re.search(r"p_brand\s*=\s*'([^']+)'", query)
    p_container_match = re.search(r"p_container\s*=\s*'([^']+)'", query)

    # Extracted values
    p_brand_value = p_brand_match.group(1) if p_brand_match else None
    p_container_value = p_container_match.group(1) if p_container_match else None

    # Construct the predicate dictionary
    predicate_dict = {
        "part": [
            {"column": "p_brand", "operator": "=", "value": f"'{p_brand_value}'", "join": False},
            {"column": "p_container", "operator": "=", "value": f"'{p_container_value}'", "join": False},
            {"column": "p_partkey", "operator": "=", "value": "l_partkey", "join": True}
        ],
        "lineitem": [
            {"column": "l_quantity", "operator": "<", "value": "subquery_result", "join": False}
        ]
    }

    return predicate_dict



def parse_tpch_query_18(query):

    # Construct the predicate dictionary
    predicate_dict = {
        "orders": [
            {"column": "o_orderkey", "operator": "=", "value": "l_orderkey", "join": True}
        ],
        "customer": [
            {"column": "c_custkey", "operator": "=", "value": "o_custkey", "join": True}
        ]
    }

    return predicate_dict



def parse_tpch_query_19(query):
  # Initialize the dictionary for predicates based on table names
    predicates = {
        "lineitem": [],
        "part": []
    }

    # Split the query into lines for easier processing
    lines = query.splitlines()

    # Initialize a list to hold the blocks
    or_blocks = []
    current_block = []

    # Read the lines and group them into OR blocks
    for line in lines:
        line = line.strip()
        if line.startswith("or") or line.startswith("("):
            if current_block:
                or_blocks.append(current_block)
                current_block = []
            if line.startswith("("):
                current_block.append(line)  # Start a new block
        elif line == ")":
            if current_block:
                current_block.append(line)  # End the current block
                or_blocks.append(current_block)
                current_block = []
        elif current_block:
            current_block.append(line)  # Add lines to the current block

    # If there's any remaining block, add it
    if current_block:
        or_blocks.append(current_block)

    # Initialize sets to hold unique values
    brands = set()
    containers = set()
    quantities_min = []
    quantities_max = []
    sizes = []
    shipmodes = set()
    shipinstructs = set()

    # Process each OR block
    for block in or_blocks:
        for line in block:
            # Extract p_brand conditions
            if "p_brand =" in line:
                brand = line.split('=')[1].strip().strip("'")
                brands.add(brand)

            # Extract p_container conditions
            if "p_container in" in line:
                containers_list = line.split('(')[1].split(')')[0].split(',')
                for container in containers_list:
                    containers.add(container.strip().strip("'"))

            # Extract l_quantity conditions
            if "l_quantity >=" in line:
                quantity_min_expr = line.split('>=')[1].strip()
                # Extract the numeric part before 'and'
                quantity_min = int(quantity_min_expr.split('and')[0].strip())
                #print(f"quantity_min: {quantity_min}")
                quantities_min.append(quantity_min)  # Store the minimum quantity

            if "l_quantity <=" in line:
                quantity_max_expr = line.split('<=')[1].strip()
                #print(f"quantity_max_expr: {quantity_max_expr}")
                # Handle arithmetic expressions by splitting on '+'
                if '+' in quantity_max_expr:
                    parts = quantity_max_expr.split('+')
                    quantity_max = sum(int(part.strip()) for part in parts)
                else:
                    quantity_max = int(quantity_max_expr.split('and')[0].strip())
                #print(f"quantity_max: {quantity_max}")    
                quantities_max.append(quantity_max)  # Store the maximum quantity

            # Extract p_size conditions
            if "p_size between" in line:
                sizes_part = line.split('between')[1].strip().split('and')
                size_min = int(sizes_part[0].strip())
                size_max = int(sizes_part[1].strip())
                sizes.append((size_min, size_max))

            # Extract l_shipmode conditions
            if "l_shipmode in" in line:
                shipmodes_list = line.split('(')[1].split(')')[0].split(',')
                for mode in shipmodes_list:
                    shipmodes.add(mode.strip().strip("'"))

            # Extract l_shipinstruct conditions
            if "l_shipinstruct =" in line:
                shipinstruct = line.split('=')[1].strip().strip("'")
                shipinstructs.add(shipinstruct)

    # Construct the predicates
    if brands:
        predicates["part"].append({
            "column": "p_brand",
            "operator": "or",
            "value": list(brands),
            "join": False
        })
    if containers:
        predicates["part"].append({
            "column": "p_container",
            "operator": "or",
            "value": list(containers),
            "join": False
        })
    if quantities_min and quantities_max:
        predicates["lineitem"].append({
            "column": "l_quantity",
            "operator": "range",
            "value": [min(quantities_min), max(quantities_max)],
            "join": False
        })
    if sizes:
        # Assuming we want to treat sizes as a range rather than individual values
        min_size = min(size[0] for size in sizes)
        max_size = max(size[1] for size in sizes)
        predicates["part"].append({
            "column": "p_size",
            "operator": "range",
            "value": [min_size, max_size],
            "join": False
        })
    if shipmodes:
        predicates["lineitem"].append({
            "column": "l_shipmode",
            "operator": "or",
            "value": list(shipmodes),
            "join": False
        })
    if shipinstructs:
        predicates["lineitem"].append({
            "column": "l_shipinstruct",
            "operator": "or",
            "value": list(shipinstructs),
            "join": False
        })

    return predicates


def parse_tpch_query_20(query):
    predicates = {
        "part": [],
        "lineitem": [],
        "supplier": [],
        "nation": []
    }

    # Split the query into lines for easier processing
    lines = query.splitlines()

    # Initialize variables to hold date values
    min_shipdate = None
    max_shipdate = None

    # Process each line to extract relevant predicates
    for line in lines:
        line = line.strip()

        # Extract p_name condition
        if "p_name like" in line:
            value = line.split("like")[1].strip().strip("'")
            predicates["part"].append({
                "column": "p_name",
                "operator": "LIKE",
                "value": value,
                "join": False
            })

        # Extract lineitem conditions
        if "l_partkey =" in line:
            value = line.split("=")[1].strip()
            predicates["lineitem"].append({
                "column": "l_partkey",
                "operator": "=",
                "value": value,
                "join": True  # This is an explicit join
            })
        if "l_suppkey =" in line:
            value = line.split("=")[1].strip()
            predicates["lineitem"].append({
                "column": "l_suppkey",
                "operator": "=",
                "value": value,
                "join": True  # This is an explicit join
            })
        if "l_shipdate >=" in line:
            # Extract the actual date from the CAST function
            min_shipdate = line.split("CAST('")[1].split("'")[0]
        if "l_shipdate <" in line:
            # Extract the actual date from the INTERVAL function
            interval_match = re.search(r"CAST\('([^']+)' AS date\) \+ INTERVAL '1 year'", line)
            if interval_match:
                base_date_str = interval_match.group(1)
                base_date = datetime.strptime(base_date_str, '%Y-%m-%d')
                max_shipdate = (base_date + relativedelta(years=1)).strftime('%Y-%m-%d')

        # Extract supplier and nation conditions
        if "s_nationkey =" in line:
            value = line.split("=")[1].strip()
            predicates["supplier"].append({
                "column": "s_nationkey",
                "operator": "=",
                "value": value,
                "join": True  # This is an explicit join
            })
        if "n_name =" in line:
            value = line.split("=")[1].strip().strip("'")
            predicates["nation"].append({
                "column": "n_name",
                "operator": "=",
                "value": value,
                "join": False
            })

    # Add a single range predicate for l_shipdate if both dates are found
    if min_shipdate and max_shipdate:
        predicates["lineitem"].append({
            "column": "l_shipdate",
            "operator": "range",
            "value": (min_shipdate, max_shipdate),
            "join": False  # Not an explicit join
        })

    return predicates


def parse_tpch_query_21(query):
    predicates = {
        "supplier": [],
        "lineitem": [],
        "orders": [],
        "nation": []
    }

    # Split the query into lines for easier processing
    lines = query.splitlines()

    # Process each line to extract relevant predicates
    for line in lines:
        line = line.strip()

        # Extract supplier and lineitem join conditions
        if "s_suppkey =" in line:
            value = line.split("=")[1].strip()
            predicates["supplier"].append({
                "column": "s_suppkey",
                "operator": "=",
                "value": value.replace("l1.", "") ,
                "join": True  # This is an explicit join
            })
        if "l1.l_suppkey =" in line:
            value = line.split("=")[1].strip().replace("l1.", "")  # Remove prefix
            predicates["lineitem"].append({
                "column": "l_suppkey",
                "operator": "=",
                "value": value.replace("l1.", ""),
                "join": True  # This is an explicit join
            })
        if "o_orderkey =" in line:
            value = line.split("=")[1].strip()
            predicates["orders"].append({
                "column": "o_orderkey",
                "operator": "=",
                "value": value.replace("l1.", "") ,
                "join": True  # This is an explicit join
            })

        # Extract filtering conditions
        if "o_orderstatus =" in line:
            value = line.split("=")[1].strip()
            predicates["orders"].append({
                "column": "o_orderstatus",
                "operator": "=",
                "value": value,
                "join": False
            })
        if "l1.l_receiptdate > l1.l_commitdate" in line:
            predicates["lineitem"].append({
                "column": "l_receiptdate",
                "operator": ">",
                "value": "l_commitdate",  # No prefix to remove
                "join": False
            })

        # Extract EXISTS and NOT EXISTS conditions
        if "exists (" in line or "not exists (" in line:
            # These indicate subqueries, we can skip adding them directly
            continue

        # Extract supplier and nation conditions
        if "s_nationkey =" in line:
            value = line.split("=")[1].strip()
            predicates["supplier"].append({
                "column": "s_nationkey",
                "operator": "=",
                "value": value,
                "join": True  # This is an explicit join
            })
        if "n_name =" in line:
            value = line.split("=")[1].strip()
            predicates["nation"].append({
                "column": "n_name",
                "operator": "=",
                "value": value,
                "join": False
            })

    return predicates

def parse_tpch_query_22(query):
    predicates = {
        "customer": [],
        "orders": []
    }

    # Split the query into lines for easier processing
    lines = query.splitlines()

    # Process each line to extract relevant predicates
    for line in lines:
        line = line.strip()

        # Extract the condition c_acctbal > value
        if "c_acctbal >" in line:
            value = line.split(">")[1].strip()  # Extract the value after '>'
            if not value.startswith("("):  # Remove leading zeros
                predicates["customer"].append({
                    "column": "c_acctbal",
                    "operator": ">",
                    "value": value,
                    "join": False
                })

        # Extract the condition o_custkey = c_custkey
        if "o_custkey =" in line:
            value = line.split("=")[1].strip()  # Extract the value after '='
            predicates["orders"].append({
                "column": "o_custkey",
                "operator": "=",
                "value": value,
                "join": True  # This is an explicit join
            })

    return predicates



def parse_tpch_query(query, template_num):
    if template_num == 1:
        return parse_tpch_query_1(query)
    elif template_num == 2:
        return parse_tpch_query_2(query)
    elif template_num == 3:
        return parse_tpch_query_3(query)
    elif template_num == 4:
        return parse_tpch_query_4(query)
    elif template_num == 5:
        return parse_tpch_query_5(query)
    elif template_num == 6:
        return parse_tpch_query_6(query)
    elif template_num == 7:
        return parse_tpch_query_7(query)
    elif template_num == 8:
        return parse_tpch_query_8(query)
    elif template_num == 9:
        return parse_tpch_query_9(query)
    elif template_num == 10:
        return parse_tpch_query_10(query)
    elif template_num == 11:
        return parse_tpch_query_11(query)
    elif template_num == 12:
        return parse_tpch_query_12(query)
    elif template_num == 13:
        return parse_tpch_query_13(query)
    elif template_num == 14:
        return parse_tpch_query_14(query)
    elif template_num == 15:
        return parse_tpch_query_15(query)
    elif template_num == 16:
        return parse_tpch_query_16(query)
    elif template_num == 17:
        return parse_tpch_query_17(query)
    elif template_num == 18:
        return parse_tpch_query_18(query)
    elif template_num == 19:
        return parse_tpch_query_19(query)
    elif template_num == 20:
        return parse_tpch_query_20(query)
    elif template_num == 21:
        return parse_tpch_query_21(query)
    elif template_num == 22:
        return parse_tpch_query_22(query)
    else:
        raise ValueError(f"Invalid template number: {template_num}")    




