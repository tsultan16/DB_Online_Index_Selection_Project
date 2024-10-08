"""
    TPC-H query generator class
"""
import pickle
import random
import math
from datetime import datetime, timedelta

# returns SSB schema as a dictionary
def get_tpch_schema():
    tables = {
        "customer": [
            ("c_custkey", "INT"),
            ("c_name", "VARCHAR(25)"),
            ("c_address", "VARCHAR(40)"),
            ("c_nationkey", "INT"),
            ("c_phone", "CHAR(15)"),
            ("c_acctbal", "DECIMAL(15, 2)"),
            ("c_mktsegment", "CHAR(10)"),
            ("c_comment", "VARCHAR(117)")
        ],
        "orders": [
            ("o_orderkey", "INT"),
            ("o_custkey", "INT"),
            ("o_orderstatus", "CHAR(1)"),
            ("o_totalprice", "DECIMAL(15, 2)"),
            ("o_orderdate", "DATE"),
            ("o_orderpriority", "CHAR(15)"),
            ("o_clerk", "CHAR(15)"),
            ("o_shippriority", "INT"),
            ("o_comment", "VARCHAR(79)")
        ],
        "lineitem": [
            ("l_orderkey", "INT"),
            ("l_partkey", "INT"),
            ("l_suppkey", "INT"),
            ("l_linenumber", "INT"),
            ("l_quantity", "DECIMAL(15, 2)"),
            ("l_extendedprice", "DECIMAL(15, 2)"),
            ("l_discount", "DECIMAL(15, 2)"),
            ("l_tax", "DECIMAL(15, 2)"),
            ("l_returnflag", "CHAR(1)"),
            ("l_linestatus", "CHAR(1)"),
            ("l_shipdate", "DATE"),
            ("l_commitdate", "DATE"),
            ("l_receiptdate", "DATE"),
            ("l_shipinstruct", "CHAR(25)"),
            ("l_shipmode", "CHAR(10)"),
            ("l_comment", "VARCHAR(44)")
        ],
        "part": [
            ("p_partkey", "INT"),
            ("p_name", "VARCHAR(55)"),
            ("p_mfgr", "CHAR(25)"),
            ("p_brand", "CHAR(10)"),
            ("p_type", "VARCHAR(25)"),
            ("p_size", "INT"),
            ("p_container", "CHAR(10)"),
            ("p_retailprice", "DECIMAL(15, 2)"),
            ("p_comment", "VARCHAR(23)")
        ],
        "supplier": [
            ("s_suppkey", "INT"),
            ("s_name", "CHAR(25)"),
            ("s_address", "VARCHAR(40)"),
            ("s_nationkey", "INT"),
            ("s_phone", "CHAR(15)"),
            ("s_acctbal", "DECIMAL(15, 2)"),
            ("s_comment", "VARCHAR(101)")
        ],
        "partsupp": [
            ("ps_partkey", "INT"),
            ("ps_suppkey", "INT"),
            ("ps_availqty", "INT"),
            ("ps_supplycost", "DECIMAL(15, 2)"),
            ("ps_comment", "VARCHAR(199)")
        ],
        "nation": [
            ("n_nationkey", "INT"),
            ("n_name", "CHAR(25)"),
            ("n_regionkey", "INT"),
            ("n_comment", "VARCHAR(152)")
        ],
        "region": [
            ("r_regionkey", "INT"),
            ("r_name", "CHAR(25)"),
            ("r_comment", "VARCHAR(152)")
        ]
    }


    pk_columns = {"customer":   ["c_custkey"],  
                  "orders":     ["o_orderkey"],  
                  "lineitem":   ["l_orderkey", "l_linenumber"],  
                  "part":       ["p_partkey"],  
                  "supplier":   ["s_suppkey"],  
                  "partsupp":   ["ps_partkey", "ps_suppkey"],  
                  "nation":     ["n_nationkey"],  
                  "region":     ["r_regionkey"]}

    return tables, pk_columns


# class for TPC-H queries
class Query:
    def __init__(self, template_id, query_string, payload, predicates, order_by, group_by):
        self.template_id = template_id
        self.query_string = query_string
        self.payload = payload
        self.predicates = predicates
        self.order_by = order_by
        self.group_by = group_by
        self.predicate_dict = None

    def __str__(self):
        return f"template id: {self.template_id}, query: {self.query_string}, payload: {self.payload}, predicates: {self.predicates}, order by: {self.order_by}, group by: {self.group_by}"    


# class for generating TPC-H queries
class TPCH_QGEN:

    def __init__(self, seed=1234, DBNAME='tpch10'):
        if seed is not None:
            random.seed(seed)

        self.DBNAME = DBNAME

        # assume table self.stats and schema available
        with open(self.DBNAME + '_stats.pkl', 'rb') as f:
            self.stats = pickle.load(f)

        self.query_templates = {
            1: """
                SELECT
                    c.c_custkey
                FROM
                    customer c
                JOIN
                    lineitem l ON c.c_custkey = l.l_orderkey 
                JOIN
                    nation n ON c.c_nationkey = n.n_nationkey
                JOIN
                    region r ON n.n_regionkey = r.r_regionkey
                WHERE
                    r.r_name = '{region}'
                    AND c.c_mktsegment = '{mktsegment}'
                    AND l.l_shipdate BETWEEN DATE '{start_date}' AND DATE '{end_date}';
            """,
            2: """
                SELECT
                    s.s_name,
                    p.p_type
                FROM
                    supplier s
                JOIN
                    partsupp ps ON s.s_suppkey = ps.ps_suppkey
                JOIN
                    part p ON ps.ps_partkey = p.p_partkey
                WHERE
                    s.s_acctbal > {acctbal_threshold}
                    AND p.p_size < {max_size}
                    AND p.p_type = '{type}'
                    AND ps.ps_availqty > {availqty_threshold}
                    AND s_name = '{supplier_name}';
            """,
            3: """
                SELECT
                    c.c_custkey,
                    c.c_name,
                    o.o_orderdate,
                    o.o_totalprice
                FROM
                    customer c
                JOIN
                    orders o ON c.c_custkey = o.o_custkey
                WHERE
                    o.o_totalprice BETWEEN {min_price} AND {max_price}
                    AND o.o_orderdate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                    AND c.c_name = '{customer_name}'
                    AND c.c_nationkey = '{nation_key}';   
            """,
            4: """
                SELECT
                    l.l_shipdate,
                    l.l_extendedprice,
                    l.l_discount,
                    l.l_quantity
                FROM
                    lineitem l
                WHERE
                    l.l_shipdate BETWEEN DATE '{start_shipdate}' AND DATE '{end_shipdate}'
                    AND l.l_extendedprice > {price_threshold}
                    AND l.l_discount < {discount_threshold}
                ORDER BY
                    l.l_shipdate;
            """,
            5: """
                SELECT
                    o_orderpriority
                FROM
                    orders
                WHERE
                    o_totalprice BETWEEN {min_price} AND {max_price}
                    AND o_orderstatus = '{status}'
                    AND o_orderdate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                    AND o_orderpriority = '{order_priority}'
                GROUP BY
                    o_orderpriority;
            """,
            6: """
                SELECT
                    p_partkey,
                    p_name,
                    l_shipmode
                FROM
                    part
                JOIN
                    lineitem ON p_partkey = l_partkey
                WHERE
                    p_retailprice BETWEEN {min_price} AND {max_price}
                    AND p_brand = '{brand}'
                    AND l_shipdate BETWEEN DATE '{start_date}' AND DATE '{end_date}';
            """,
            7: """
                SELECT 
                    l.l_orderkey, 
                    o.o_orderdate, 
                    o.o_shippriority
                FROM 
                    lineitem l
                JOIN 
                    orders o ON l.l_orderkey = o.o_orderkey
                WHERE 
                    o.o_orderdate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                    AND l.l_discount BETWEEN {min_discount} AND {max_discount}
                    AND l.l_quantity BETWEEN {min_quantity} AND {max_quantity}
                ORDER BY 
                    o.o_orderdate;
            """,
            8: """
                SELECT 
                    l.l_extendedprice, 
                    s.s_suppkey, 
                    s.s_name
                FROM 
                    lineitem l
                JOIN 
                    supplier s ON l.l_suppkey = s.s_suppkey
                WHERE 
                    l.l_shipdate BETWEEN  DATE '{start_shipdate}' AND DATE '{end_shipdate}'
                    AND s.s_name = '{supplier_name}'
                ORDER BY 
                    l.l_shipdate;
            """,
            9: """
                SELECT 
                    s.s_name, 
                    ps.ps_availqty, 
                    o.o_orderdate
                FROM 
                    supplier s
                JOIN 
                    partsupp ps ON s.s_suppkey = ps.ps_suppkey
                JOIN 
                    lineitem l ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
                JOIN 
                    orders o ON l.l_orderkey = o.o_orderkey
                WHERE 
                    o.o_orderdate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                    AND s.s_name = '{supplier_name}'
                ORDER BY 
                    o.o_orderdate;
            """,
            10: """
                SELECT 
                    ps.ps_partkey, 
                    ps.ps_suppkey, 
                    ps.ps_availqty, 
                    ps.ps_supplycost
                FROM 
                    partsupp ps
                WHERE 
                    ps.ps_availqty BETWEEN {min_quantity} AND {max_quantity}
                    AND ps.ps_supplycost BETWEEN {min_cost} AND {max_cost}
                ORDER BY 
                    ps.ps_supplycost;
            """,
            11: """
                SELECT
                    ps_partkey,
                    ps_suppkey,
                    SUM(ps_supplycost * ps_availqty) AS total_inventory_cost
                FROM
                    partsupp
                WHERE
                    ps_availqty > {quantity_threshold}
                GROUP BY
                    ps_partkey,
                    ps_suppkey
                ORDER BY
                    total_inventory_cost DESC;
            """,
            12: """
                SELECT
                    p_name,
                    s_name,
                    ps_availqty
                FROM
                    partsupp
                JOIN
                    part ON ps_partkey = p_partkey
                JOIN
                    supplier ON ps_suppkey = s_suppkey
                WHERE
                    p_mfgr = '{manufacturer}'
                    AND ps_supplycost < {cost_threshold}
                ORDER BY
                    ps_availqty DESC;
            """,
            13: """
                SELECT
                    ps_partkey
                FROM
                    partsupp
                WHERE
                    ps_supplycost BETWEEN {min_cost} AND {max_cost};
            """,
            14: """
                SELECT
                    s_name,
                    SUM(ps_availqty) AS total_quantity
                FROM
                    supplier
                JOIN
                    partsupp ON s_suppkey = ps_suppkey
                WHERE
                    s_nationkey = {nation_key}
                GROUP BY
                    s_name
                ORDER BY
                    total_quantity DESC;
            """,
            15: """
                SELECT
                    ps_partkey,
                    p_type,
                    AVG(ps_supplycost) AS avg_cost
                FROM
                    partsupp
                JOIN
                    part ON ps_partkey = p_partkey
                WHERE
                    p_size = {size}
                GROUP BY
                    ps_partkey,
                    p_type
                ORDER BY
                    avg_cost ASC;
            """,
            16: """
                SELECT
                    c.c_custkey,
                    c.c_name,
                    o.o_orderdate,
                    s.s_name,
                    l.l_orderkey
                FROM
                    customer c
                JOIN
                    orders o ON c.c_custkey = o.o_custkey
                JOIN
                    lineitem l ON o.o_orderkey = l.l_orderkey
                JOIN
                    supplier s ON l.l_suppkey = s.s_suppkey
                WHERE
                    c.c_acctbal > {acctbal_threshold}
                    AND s.s_acctbal < {supplier_acctbal_threshold}
                    AND o.o_orderpriority = '{order_priority}'
                    AND l.l_shipmode = '{ship_mode}'
                    AND o.o_orderdate BETWEEN DATE '{start_date}' AND DATE '{end_date}';
            """,
        }

        self.predicates = {
            1: {
                'lineitem': ["l_shipdate", "l_orderkey"],
                'region': ["r_name", "r_regionkey"],
                'customer': ["c_mktsegment", "c_nationkey", "c_custkey"],
                'nation': ["n_nationkey", "n_regionkey"]
            },
            2: {
                'supplier': ["s_acctbal", "s_suppkey", "s_name"],
                'part': ["p_size", "p_type", "p_partkey"],
                'partsupp': ["ps_partkey", "ps_suppkey", "ps_availqty"]
            },
            3: {
                'customer': ["c_nationkey", "c_name", "c_custkey"],
                'orders': ["o_totalprice", "o_orderdate", "o_custkey"],
            },
            4: {
                'lineitem': ["l_shipdate", "l_extendedprice", "l_discount"]
            },
            5: {
                'orders': ["o_totalprice", "o_orderstatus", "o_orderpriority", "o_orderdate"]
            },
            6: {
                'part': ["p_retailprice", "p_brand"],
                'lineitem': ["l_shipdate", "l_partkey"]
            },
            7: {
                'lineitem': ["l_discount", "l_orderkey", "l_quantity"],
                'orders': ["o_orderdate", "o_orderkey"],
            },
            8: {
                'lineitem': ["l_shipdate", "l_suppkey"],
                'supplier': ["s_name", "s_suppkey"]
            },
            9: {
                'lineitem': ["l_partkey", "l_suppkey", "l_orderkey"],
                'supplier': ["s_name"],
                'partsupp': ["ps_partkey", "ps_suppkey"],
                'orders': ["o_orderdate", "o_orderkey"]
            },
            10: {
                'partsupp': ["ps_availqty", "ps_supplycost"], 
            },
            11: {
                'partsupp': ["ps_availqty"]
            },
            12: {
                'part': ["p_mfgr"],
                'partsupp': ["ps_supplycost"]
            },
            13: {
                'partsupp': ["ps_supplycost"]
            },
            14: {
                'supplier': ["s_nationkey"]
            },
            15: {
                'part': ["p_size"]
            },
            16: {
                'customer': ["c_acctbal"],
                'supplier': ["s_acctbal"],
                'orders': ["o_orderpriority", "o_orderdate"],
                'lineitem': ["l_shipmode"]
            }
        }

        self.payloads = {
            1: {
                "customer": ["c_custkey"]
            },
            2: {
                "supplier": ["s_name"],
                "part": ["p_type"]
            },
            3: {
                "customer": ["c_custkey", "c_name"],
                "orders": ["o_orderdate", "o_totalprice"]
            },
            4: {
                "lineitem": ["l_shipdate", "l_extendedprice", "l_discount", "l_quantity"],
            },
            5: {
                "orders": ["o_orderpriority"]
            },
            6: {
                "part": ["p_partkey", "p_name"],
                "lineitem": ["l_shipmode"]
            },
            7: {
                "lineitem": ["l_orderkey"],
                "orders": ["o_orderdate", "o_shippriority"]
            },
            8: {
                "lineitem": ["l_extendedprice"],
                "supplier": ["s_name", "s_suppkey"]
            },
            9: {
                "orders": ["o_orderdate"],
                "partsupp": ["ps_availqty"],
                "supplier": ["s_name"]
            },
            10: {
                "partsupp": ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"]
            },
            11: {
                "partsupp": ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"]
            },
            12: {
                "part": ["p_name"],
                "supplier": ["s_name"],
                "partsupp": ["ps_availqty"]
            },
            13: {
                "partsupp": ["ps_partkey"]
            },
            14: {
                "supplier": ["s_name"],
                "partsupp": ["ps_availqty"]
            },
            15: {
                "partsupp": ["ps_partkey", "ps_supplycost"],
                "part": ["p_type"]
            },
            16: {
                "customer": ["c_custkey", "c_name"],
                "orders": ["o_orderdate"],
                "supplier": ["s_name"],
                "lineitem": ["l_orderkey"]
            }
        }

        self.order_bys = {
            1: {},
            2: {},
            3: {},
            4: {
                "lineitem": ["l_shipdate"]
            },
            5: {},
            6: {},
            7: { 
                "orders": ["o_orderdate"]
            },
            8: { "lineitem": ["l_shipdate"] 
                },
            9: {
                "orders": ["o_orderdate"]   
            },
            10: {
                "partsupp": ["ps_supplycost"]
            },
            11: {
                "": ["total_inventory_cost"]
            },
            12: {
                "": ["ps_availqty"]
            },
            13: {},
            14: {
                "": ["total_quantity"]
            },
            15: {
                "": ["avg_cost"]
            },
            16: {}
        }

        self.group_bys = {
            1: {},
            2: {},
            3: {},
            4: {},
            5: {
                "orders": ["o_orderpriority"]
            },
            6: {},
            7: {},
            8: {},
            9: {},
            10: {},
            11: {
                "partsupp": ["ps_partkey", "ps_suppkey"]
            },
            12: {},
            13: {},
            14: {
                "supplier": ["s_name"]
            },
            15: {
                "partsupp": ["ps_partkey"],
                "part": ["p_type"]
            },
            16: {}
        }


        self.predicate_dicts = {
            1: {
                "region": [
                    {"column": "r_name", "operator": "eq", "value": "'{region}'", "join": False}, 
                    {"column": "r_regionkey", "operator": "eq", "value": "n_regionkey", "join": True}

                ],
                "lineitem": [
                    {"column": "l_shipdate", "operator": "range", "value": ("DATE '{start_date}'", "DATE '{end_date}'"), "join": False},
                    {"column": "l_orderkey", "operator": "eq", "value": "c_custkey", "join": True}

                ],
                "customer": [
                    {"column": "c_mktsegment", "operator": "eq", "value": "{mktsegment}", "join": False},
                    {"column": "c_nationkey", "operator": "eq", "value": "n_nationkey", "join": True},
                    {"column": "c_custkey", "operator": "eq", "value": "l_orderkey", "join": True},
                ],
                "nation": [
                    {"column": "n_nationkey", "operator": "eq", "value": "c_nationkey", "join": True},
                    {"column": "n_regionkey", "operator": "eq", "value": "r_regionkey", "join": True}
                ]
            },
            2: {
                "supplier": [
                    {"column": "s_acctbal", "operator": ">", "value": "{acctbal_threshold}", "join": False},
                    {"column": "s_name", "operator": "eq", "value": "'{supplier_name}'", "join": False},
                    {"column": "s_suppkey", "operator": "eq", "value": "ps_suppkey", "join": True}
                ],
                "part": [
                    {"column": "p_size", "operator": "range", "value": "{min_size} AND {max_size}", "join": False},
                    {"column": "p_type", "operator": "eq", "value": "'{type}'", "join": False},
                    {"column": "p_partkey", "operator": "eq", "value": "ps_partkey", "join": True}
                ],
                "partsupp": [
                    {"column": "ps_availqty", "operator": ">", "value": "{availqty_threshold}", "join": False},
                    {"column": "ps_partkey", "operator": "eq", "value": "p_partkey", "join": True},
                    {"column": "ps_suppkey", "operator": "eq", "value": "s_suppkey", "join": True}
                ]
            },
            3: {
                "customer": [
                    {"column": "c_name", "operator": "eq", "value": "'{customer_name}'", "join": False},
                    {"column": "c_nationkey", "operator": "eq", "value": "{nation_key}", "join": False},
                    {"column": "c_custkey", "operator": "eq", "value": "o_custkey", "join": True},
                ],
                "orders": [
                    {"column": "o_totalprice", "operator": "range", "value": "{min_price} AND {max_price}", "join": False},
                    {"column": "o_orderdate", "operator": "range", "value": "DATE '{start_date}' AND DATE '{end_date}'", "join": False},
                    {"column": "o_custkey", "operator": "eq", "value": "c_custkey", "join": True},
                ]
            },
            4: {
                "lineitem": [{"column": "l_shipdate", "operator": "range", "value": "DATE '{start_shipdate}' AND DATE '{end_shipdate}'", "join": False},
                             {"column": "l_extendedprice", "operator": ">", "value": "{price_threshold}", "join": False},
                             {"column": "l_discount", "operator": "<", "value": "{discount_threshold}", "join": False}]     
            },
            5: {
                "orders": [
                    {"column": "o_totalprice", "operator": "range", "value": "{min_price} AND {max_price}", "join": False},
                    {"column": "o_orderstatus", "operator": "eq", "value": "'{status}'", "join": False},
                    {"column": "o_orderdate", "operator": "range", "value": "DATE '{start_date}' AND DATE '{end_date}'", "join": False},
                    {"column": "o_orderpriority", "operator": "eq", "value": "{order_priority}", "join": False}
                ]
            },
            6: {
                "part": [
                    {"column": "p_retailprice", "operator": "range", "value": "{min_price} AND {max_price}", "join": False},
                    {"column": "p_brand", "operator": "eq", "value": "'{brand}'", "join": False}
                ],
                "lineitem": [
                    {"column": "l_partkey", "operator": "eq", "value": "p_partkey", "join": True},
                    {"column": "l_shipdate", "operator": "range", "value": "DATE '{start_date}' AND DATE '{end_date}'", "join": False}
                ]
            },
            7: {
                "lineitem": [{"column": "l_discount", "operator": "range", "value": "{min_discount} AND {max_discount}", "join": False},
                                {"column": "l_quantity", "operator": "range", "value": "{min_quantity} AND {max_quantity}", "join": False},
                                {"column": "l_orderkey", "operator": "eq", "value": "o_orderkey", "join": True}],
                "orders": [{"column": "o_orderdate", "operator": "range", "value": "DATE '{start_date}' AND DATE '{end_date}'", "join": False},
                               {"column": "o_orderkey", "operator": "eq", "value": "l_orderkey", "join": True}]               
            },
            8: {
                "lineitem": [{"column": "l_shipdate", "operator": "range", "value": "DATE '{start_shipdate}' AND DATE '{end_shipdate}'", "join": False},
                                {"column": "l_suppkey", "operator": "eq", "value": "s_suppkey", "join": True}],
                "supplier": [{"column": "s_name", "operator": "eq", "value": "'{supplier_name}'", "join": False},
                                {"column": "s_suppkey", "operator": "eq", "value": "ps_suppkey", "join": True}]
            },
            9: {
                "lineitem": [
                    {"column": "l_orderkey", "operator": "eq", "value": "o_orderkey", "join": True},
                    {"column": "l_partkey", "operator": "eq", "value": "ps_partkey", "join": True},
                    {"column": "l_suppkey", "operator": "eq", "value": "ps_suppkey", "join": True}
                ],
                "supplier": [
                    {"column": "s_name", "operator": "eq", "value": "'{supplier_name}'", "join": False},
                    {"column": "s_suppkey", "operator": "eq", "value": "ps_suppkey", "join": True}
                ],
                "partsupp": [
                    {"column": "ps_partkey", "operator": "eq", "value": "l_partkey", "join": True},
                    {"column": "ps_suppkey", "operator": "eq", "value": "l_suppkey", "join": True}
                ],
                "orders": [
                    {"column": "o_orderdate", "operator": "range", "value": "DATE '{start_date}' AND DATE '{end_date}'", "join": False},
                    {"column": "o_custkey", "operator": "eq", "value": "l_orderkey", "join": True}
                ]
            },
            10: {
                "partsupp": [{"column": "ps_availqty", "operator": "range", "value": "{min_quantity} AND {max_quantity}", "join": False},
                             {"column": "ps_supplycost", "operator": "range", "value": "{min_cost} AND {max_cost}", "join": False}]
            },
            11: {
                "partsupp": [
                    {"column": "ps_availqty", "operator": ">", "value": "{quantity_threshold}", "join": False}
                ]
            },
            12: {
                "part": [
                    {"column": "p_mfgr", "operator": "eq", "value": "'{manufacturer}'", "join": False},
                    {"column": "p_partkey", "operator": "eq", "value": "ps_partkey", "join": True}
                ],
                "partsupp": [
                    {"column": "ps_supplycost", "operator": "<", "value": "{cost_threshold}", "join": False},
                    {"column": "ps_suppkey", "operator": "eq", "value": "s_suppkey", "join": True},
                    {"column": "ps_partkey", "operator": "eq", "value": "p_partkey", "join": True}
                ],
                "supplier": [
                    {"column": "s_suppkey", "operator": "eq", "value": "ps_suppkey", "join": True}
                ]
            },
            13: {
                "partsupp": [
                    {"column": "ps_supplycost", "operator": "range", "value": "{min_cost} AND {max_cost}", "join": False}
                ]
            },
            14: {
                "supplier": [
                    {"column": "s_nationkey", "operator": "eq", "value": "{nation_key}", "join": False},
                    {"column": "s_suppkey", "operator": "eq", "value": "ps_suppkey", "join": True}
                ],
                "partsupp": [
                    {"column": "ps_partkey", "operator": "eq", "value": "p_partkey", "join": True}
                ]
            },
            15: {
                "part": [
                    {"column": "p_size", "operator": "eq", "value": "{size}", "join": False},
                    {"column": "p_partkey", "operator": "eq", "value": "ps_partkey", "join": True}
                ]
            },
            16: {
                "customer": [
                    {"column": "c_acctbal", "operator": ">", "value": "{acctbal_threshold}", "join": False},
                    {"column": "c_custkey", "operator": "eq", "value": "o_custkey", "join": True},
                ],
                "supplier": [
                    {"column": "s_acctbal", "operator": "<", "value": "{supplier_acctbal_threshold}", "join": False}
                ],
                "orders": [
                    {"column": "o_orderpriority", "operator": "eq", "value": "'{order_priority}'", "join": False},
                    {"column": "o_orderdate", "operator": "range", "value": "DATE '{start_date}' AND DATE '{end_date}'", "join": False},
                    {"column": "o_orderkey", "operator": "eq", "value": "l_orderkey", "join": True}
                ],
                "lineitem": [
                    {"column": "l_shipmode", "operator": "eq", "value": "'{ship_mode}'", "join": False},
                    {"column": "l_suppkey", "operator": "eq", "value": "s_suppkey", "join": True}
                ]
            }
        }


    def generate_query(self, template_num):
        if template_num not in self.query_templates:
            raise ValueError("Template not found")

        template = self.query_templates[template_num]

        # Fill parameters based on statistics
        if template_num == 1:
            # Choose a random region name from the region statistics
            region = random.choice(list(self.stats['region']['r_name']['histogram'].keys()))

            # choose a random customer mktsegment
            mktsegment = random.choice(list(self.stats['customer']['c_mktsegment']['histogram'].keys()))
            
            # Determine a random start date within the available shipdate range
            min_shipdate = self.stats['lineitem']['l_shipdate']['min']
            max_shipdate = self.stats['lineitem']['l_shipdate']['max']
            start_date = min_shipdate + timedelta(days=random.randint(0, (max_shipdate - min_shipdate).days))
            
            # Define an end date up to 30 days after the start date
            end_date = start_date + timedelta(days=random.randint(1, 30))
            
            # Format the query with the selected parameters
            query = template.format(region=region, mktsegment=mktsegment, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            
            # Update predicate dicts to reflect the selected parameters
            self.predicate_dicts[template_num]['region'][0]['value'] = f"'{region}'"
            self.predicate_dicts[template_num]['customer'][0]['value'] = mktsegment
            self.predicate_dicts[template_num]['lineitem'][0]['value'] = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        elif template_num == 2:
            # Choose a realistic account balance threshold within supplier's account balance range
            acctbal_threshold = random.uniform(float(self.stats['supplier']['s_acctbal']['min']), float(self.stats['supplier']['s_acctbal']['max']))
            # choose a random supplier name
            supplier_name = "Supplier#0000" +  f"{random.randint(1, 10000):05}"
            # Choose a realistic size range for parts
            min_size = random.randint(self.stats['part']['p_size']['min'], self.stats['part']['p_size']['max'] - 1)
            max_size = random.randint(min_size + 1, self.stats['part']['p_size']['max'])
            # Choose a random part type
            part_type = random.choice(list(self.stats['part']['p_type']['histogram'].keys()))
            # Choose a realistic quantity threshold within the available quantity range
            availqty_threshold = random.randint(1, self.stats['partsupp']['ps_availqty']['max'])

            query = template.format(acctbal_threshold=acctbal_threshold, supplier_name=supplier_name, min_size=min_size, max_size=max_size, type=part_type, availqty_threshold=availqty_threshold)
            self.predicate_dicts[template_num]['supplier'][0]['value'] = acctbal_threshold
            self.predicate_dicts[template_num]['supplier'][1]['value'] = supplier_name
            self.predicate_dicts[template_num]['part'][0]['value'] = (min_size, max_size)
            self.predicate_dicts[template_num]['part'][1]['value'] = part_type
            self.predicate_dicts[template_num]['partsupp'][0]['value'] = availqty_threshold


        elif template_num == 3:
            # choose a random customer name
            customer_name = "Customer#000" +  f"{random.randint(1, 150000):06}"
            # Choose a random nation key
            nation_key = random.choice(list(self.stats['customer']['c_nationkey']['histogram'].keys()))
            # Choose a realistic price range for orders
            min_price = float(self.stats['orders']['o_totalprice']['min'])
            max_price = float(self.stats['orders']['o_totalprice']['max'])
            random_min_price = random.uniform(min_price, max_price - 1000)
            random_max_price = random_min_price + random.uniform(10, 200)
            # Determine a random start date within the available order date range
            min_orderdate = self.stats['orders']['o_orderdate']['min']
            max_orderdate = self.stats['orders']['o_orderdate']['max']
            start_date = min_orderdate + timedelta(days=random.randint(0, (max_orderdate - min_orderdate).days)) 
            # Define an end date up to 5 days after the start date
            end_date = start_date + timedelta(days=random.randint(1, 5))

            query = template.format(customer_name=customer_name, nation_key=nation_key, min_price=random_min_price, max_price=random_max_price, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            self.predicate_dicts[template_num]['customer'][0]['value'] = customer_name
            self.predicate_dicts[template_num]['customer'][1]['value'] = nation_key
            self.predicate_dicts[template_num]['orders'][0]['value'] = (random_min_price, random_max_price)
            self.predicate_dicts[template_num]['orders'][1]['value'] = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            

        elif template_num == 4:
            # Select a realistic shipdate range
            min_shipdate = self.stats['lineitem']['l_shipdate']['min']
            max_shipdate = self.stats['lineitem']['l_shipdate']['max']
            start_shipdate = min_shipdate + timedelta(days=random.randint(0, (max_shipdate - min_shipdate).days - 3))
            end_shipdate = start_shipdate + timedelta(days=random.randint(1, 3))
            # Select a realistic price threshold
            price_threshold = random.uniform(float(self.stats['lineitem']['l_extendedprice']['min']), float(self.stats['lineitem']['l_extendedprice']['max']))
            # Select a realistic discount threshold
            discount_threshold = random.uniform(float(self.stats['lineitem']['l_discount']['min']), float(self.stats['lineitem']['l_discount']['max']))

            query = template.format(start_shipdate=start_shipdate.strftime('%Y-%m-%d'), end_shipdate=end_shipdate.strftime('%Y-%m-%d'), price_threshold=price_threshold, discount_threshold=discount_threshold)
            self.predicate_dicts[template_num]['lineitem'][0]['value'] = (start_shipdate.strftime('%Y-%m-%d'), end_shipdate.strftime('%Y-%m-%d'))
            self.predicate_dicts[template_num]['lineitem'][1]['value'] = price_threshold
            self.predicate_dicts[template_num]['lineitem'][2]['value'] = discount_threshold


        elif template_num == 5:
            # choose a realistic price range for orders
            min_price = float(self.stats['orders']['o_totalprice']['min'])
            max_price = float(self.stats['orders']['o_totalprice']['max'])
            random_min_price = random.uniform(min_price, max_price - 1000)
            random_max_price = random_min_price + random.uniform(100, 1000)
            # choose a random order status            
            status = random.choice(list(self.stats['orders']['o_orderstatus']['histogram'].keys()))
            start_date = self.stats['orders']['o_orderdate']['min'] + timedelta(days=random.randint(0, (self.stats['orders']['o_orderdate']['max'] - self.stats['orders']['o_orderdate']['min']).days))
            end_date = start_date + timedelta(days=random.randint(1, 5))
            # choose a random order priority
            order_priority = random.choice(list(self.stats['orders']['o_orderpriority']['histogram'].keys()))

            query = template.format(min_price=min_price, max_price=max_price, status=status, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'), order_priority=order_priority)
            self.predicate_dicts[template_num]['orders'][0]['value'] = (random_min_price, random_max_price)
            self.predicate_dicts[template_num]['orders'][1]['value'] = status
            self.predicate_dicts[template_num]['orders'][2]['value'] =  (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            self.predicate_dicts[template_num]['orders'][3]['value'] = order_priority


        elif template_num == 6:
            min_price = float(self.stats['part']['p_retailprice']['min'])
            max_price = float(self.stats['part']['p_retailprice']['max'])
            random_min_price = random.uniform(min_price, max_price - 100)
            random_max_price = random_min_price + random.uniform(1.0, 20.0) #random.uniform(random_min_price + 1, max_price)
            brand = random.choice(list(self.stats['part']['p_brand']['histogram'].keys()))
            start_date = self.stats['lineitem']['l_shipdate']['min'] + timedelta(days=random.randint(0, (self.stats['lineitem']['l_shipdate']['max'] - self.stats['lineitem']['l_shipdate']['min']).days))
            end_date = start_date + timedelta(days=random.randint(1, 3))
            query = template.format(min_price=random_min_price, max_price=random_max_price, brand=brand, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            self.predicate_dicts[template_num]['part'][0]['value'] = (random_min_price, random_max_price)
            self.predicate_dicts[template_num]['part'][1]['value'] = brand
            self.predicate_dicts[template_num]['lineitem'][1]['value'] = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        elif template_num == 7:
            # Select a realistic discount range
            min_discount = float(self.stats['lineitem']['l_discount']['min'])
            max_discount = float(self.stats['lineitem']['l_discount']['max'])
            random_min_discount = random.uniform(min_discount, max_discount - 0.02)
            random_max_discount = random_min_discount + random.uniform(0.01, 0.02)

            # Select a realistic quantity range
            min_quantity = float(self.stats['lineitem']['l_quantity']['min'])
            max_quantity = float(self.stats['lineitem']['l_quantity']['max'])
            random_min_quantity = random.uniform(min_quantity, max_quantity - 10)
            random_max_quantity =  random_min_quantity + random.uniform(1.0, 3.0)

            # Select a realistic order date range
            start_date = self.stats['orders']['o_orderdate']['min'] + timedelta(days=random.randint(0, (self.stats['orders']['o_orderdate']['max'] - self.stats['orders']['o_orderdate']['min']).days))
            end_date = start_date + timedelta(days=random.randint(1, 5))

            query = template.format(min_discount=random_min_discount, max_discount=random_max_discount, min_quantity=random_min_quantity, max_quantity=random_max_quantity, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))

            self.predicate_dicts[template_num]['lineitem'][0]['value'] = (random_min_discount, random_max_discount)
            self.predicate_dicts[template_num]['lineitem'][1]['value'] = (random_min_quantity, random_max_quantity)
            self.predicate_dicts[template_num]['orders'][0]['value'] = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))


        elif template_num == 8:
            # Select a realistic shipdate range
            shipdate_min = self.stats['lineitem']['l_shipdate']['min']
            shipdate_max = self.stats['lineitem']['l_shipdate']['max']
            start_shipdate = shipdate_min + timedelta(days=random.randint(0, (shipdate_max - shipdate_min).days - 30))
            end_shipdate = start_shipdate + timedelta(days=random.randint(1, 30))

            # Select a random supplier name
            supplier_name = "Supplier#0000" +  f"{random.randint(1, 10000):05}"
            
            query = template.format(start_shipdate=start_shipdate.strftime('%Y-%m-%d'), end_shipdate=end_shipdate.strftime('%Y-%m-%d'), supplier_name=supplier_name)

            self.predicate_dicts[template_num]['lineitem'][0]['value'] = (start_shipdate.strftime('%Y-%m-%d'), end_shipdate.strftime('%Y-%m-%d'))
            self.predicate_dicts[template_num]['supplier'][0]['value'] = supplier_name

        elif template_num == 9:
            # Select a realistic order date range
            orderdate_min = self.stats['orders']['o_orderdate']['min']
            orderdate_max = self.stats['orders']['o_orderdate']['max']
            start_orderdate = orderdate_min + timedelta(days=random.randint(0, (orderdate_max - orderdate_min).days - 30))
            end_orderdate = start_orderdate + timedelta(days=random.randint(1, 10))

            # Select a random supplier name
            supplier_name = "Supplier#0000" +  f"{random.randint(1, 10000):05}"

            query = template.format(start_date=start_orderdate.strftime('%Y-%m-%d'), end_date=end_orderdate.strftime('%Y-%m-%d'), supplier_name=supplier_name)

            self.predicate_dicts[template_num]['orders'][0]['value'] = (start_orderdate.strftime('%Y-%m-%d'), end_orderdate.strftime('%Y-%m-%d'))
            self.predicate_dicts[template_num]['supplier'][0]['value'] = supplier_name

        elif template_num == 10:
            # Select a realistic quantity range
            min_quantity = int(self.stats['partsupp']['ps_availqty']['min'])
            max_quantity = int(self.stats['partsupp']['ps_availqty']['max'])
            random_min_quantity = random.randint(min_quantity, max_quantity - 10)
            random_max_quantity = random_min_quantity + random.randint(1, 10)

            # Select a realistic cost range
            min_cost = float(self.stats['partsupp']['ps_supplycost']['min'])
            max_cost = float(self.stats['partsupp']['ps_supplycost']['max'])
            random_min_cost = random.uniform(min_cost, max_cost - 5)
            random_max_cost = random_min_cost + random.uniform(1, 5)

            query = template.format(min_quantity=random_min_quantity, max_quantity=random_max_quantity, min_cost=random_min_cost, max_cost=random_max_cost)

            self.predicate_dicts[template_num]['partsupp'][0]['value'] = (random_min_quantity, random_max_quantity)
            self.predicate_dicts[template_num]['partsupp'][1]['value'] = (random_min_cost, random_max_cost)

        elif template_num == 11:
            quantity_threshold = random.randint(int(self.stats['partsupp']['ps_availqty']['min']), int(self.stats['partsupp']['ps_availqty']['max']))
            query = template.format(quantity_threshold=quantity_threshold)
            self.predicate_dicts[template_num]['partsupp'][0]['value'] = quantity_threshold

        elif template_num == 12:
            manufacturer = random.choice(list(self.stats['part']['p_mfgr']['histogram'].keys()))
            cost_threshold = random.uniform(float(self.stats['partsupp']['ps_supplycost']['min']), float(self.stats['partsupp']['ps_supplycost']['max']))
            query = template.format(manufacturer=manufacturer, cost_threshold=cost_threshold)
            self.predicate_dicts[template_num]['part'][0]['value'] = manufacturer
            self.predicate_dicts[template_num]['partsupp'][0]['value'] = cost_threshold

        elif template_num == 13:
            min_cost = random.uniform(float(self.stats['partsupp']['ps_supplycost']['min']), float(self.stats['partsupp']['ps_supplycost']['max']) - 500)
            max_cost = random.uniform(min_cost + 1, float(self.stats['partsupp']['ps_supplycost']['max']))
            query = template.format(min_cost=min_cost, max_cost=max_cost)
            self.predicate_dicts[template_num]['partsupp'][0]['value'] = (min_cost, max_cost)

        elif template_num == 14:
            nation_key = random.choice(list(self.stats['nation']['n_nationkey']['histogram'].keys()))
            query = template.format(nation_key=nation_key)
            self.predicate_dicts[template_num]['supplier'][0]['value'] = nation_key

        elif template_num == 15:
            size = random.choice(list(self.stats['part']['p_size']['histogram'].keys()))
            query = template.format(size=size)
            self.predicate_dicts[template_num]['part'][0]['value'] = size

        elif template_num == 16:
            acctbal_threshold = random.uniform(float(self.stats['customer']['c_acctbal']['min']), float(self.stats['customer']['c_acctbal']['max']))
            supplier_acctbal_threshold = random.uniform(float(self.stats['supplier']['s_acctbal']['min']), float(self.stats['supplier']['s_acctbal']['max']))
            order_priority = random.choice(list(self.stats['orders']['o_orderpriority']['histogram'].keys()))
            ship_mode = random.choice(list(self.stats['lineitem']['l_shipmode']['histogram'].keys()))
            start_date = self.stats['orders']['o_orderdate']['min'] + timedelta(days=random.randint(0, (self.stats['orders']['o_orderdate']['max'] - self.stats['orders']['o_orderdate']['min']).days))
            end_date = start_date + timedelta(days=random.randint(1, 30))
            query = template.format(acctbal_threshold=acctbal_threshold, supplier_acctbal_threshold=supplier_acctbal_threshold, order_priority=order_priority, ship_mode=ship_mode, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            
            self.predicate_dicts[template_num]['customer'][0]['value'] = acctbal_threshold
            self.predicate_dicts[template_num]['supplier'][0]['value'] = supplier_acctbal_threshold
            self.predicate_dicts[template_num]['orders'][0]['value'] = order_priority
            self.predicate_dicts[template_num]['lineitem'][0]['value'] = ship_mode
            self.predicate_dicts[template_num]['orders'][1]['value'] = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))


        else:
            raise ValueError("Query template not implemented for this template number.")
    

        # create Query object
        query = Query(template_num, query, self.payloads[template_num], self.predicates[template_num], self.order_bys[template_num], self.group_bys[template_num])
        query.predicate_dict = dict(self.predicate_dicts[template_num]) 
    
        return query
























