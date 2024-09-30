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

    def __init__(self, seed=1234):
        if seed is not None:
            random.seed(seed)

        self.DBNAME = 'tpch4'

        # assume table self.stats and schema available
        with open(self.DBNAME + '_stats.pkl', 'rb') as f:
            self.stats = pickle.load(f)

        self.query_templates = {
            1: """
                SELECT
                    c_custkey,
                    o_orderdate,
                    SUM(l_extendedprice * (1 - l_discount)) AS total_spent
                FROM
                    customer
                JOIN
                    orders ON c_custkey = o_custkey
                JOIN
                    lineitem ON o_orderkey = l_orderkey
                WHERE
                    c_region = '{region}'
                    AND l_shipdate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                GROUP BY
                    c_custkey,
                    o_orderdate
                ORDER BY
                    total_spent DESC;
            """,
            2: """
                SELECT
                    s_suppkey,
                    s_name,
                    p_type,
                    SUM(ps_supplycost) AS total_cost
                FROM
                    supplier
                JOIN
                    partsupp ON s_suppkey = ps_suppkey
                JOIN
                    part ON ps_partkey = p_partkey
                WHERE
                    s_acctbal > {acctbal_threshold}
                    AND p_size BETWEEN {min_size} AND {max_size}
                GROUP BY
                    s_suppkey,
                    s_name,
                    p_type
                ORDER BY
                    total_cost ASC;
            """,
            3: """
                SELECT
                    n_name,
                    COUNT(DISTINCT c_custkey) AS num_customers
                FROM
                    nation
                JOIN
                    customer ON n_nationkey = c_nationkey
                WHERE
                    c_acctbal < {acctbal_threshold}
                GROUP BY
                    n_name
                ORDER BY
                    num_customers DESC;
            """,
            4: """
                SELECT
                    r_name,
                    p_mfgr,
                    AVG(l_quantity) AS avg_quantity
                FROM
                    region
                JOIN
                    nation ON r_regionkey = n_regionkey
                JOIN
                    supplier ON n_nationkey = s_nationkey
                JOIN
                    lineitem ON s_suppkey = l_suppkey
                JOIN
                    part ON l_partkey = p_partkey
                WHERE
                    r_name LIKE '{region_pattern}'
                GROUP BY
                    r_name,
                    p_mfgr
                ORDER BY
                    avg_quantity DESC;
            """,
            5:"""
                SELECT
                    o_orderpriority,
                    COUNT(*) AS order_count
                FROM
                    orders
                WHERE
                    o_totalprice > {price_threshold}
                    AND o_orderstatus = '{status}'
                    AND o_orderdate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                    AND o_orderpriority IN ('1-URGENT', '2-HIGH')
                GROUP BY
                    o_orderpriority
                ORDER BY
                    order_count DESC;
            """,
            6: """
                SELECT
                    p_partkey,
                    p_name,
                    l_shipmode,
                    COUNT(*) AS ship_count
                FROM
                    part
                JOIN
                    lineitem ON p_partkey = l_partkey
                WHERE
                    p_retailprice BETWEEN {min_price} AND {max_price}
                    AND p_brand = '{brand}'
                    AND l_shipdate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                GROUP BY
                    p_partkey,
                    p_name,
                    l_shipmode
                ORDER BY
                    ship_count DESC;
            """,
            7: """
                SELECT
                    c_name,
                    o_orderstatus,
                    SUM(l_extendedprice) AS revenue
                FROM
                    customer
                JOIN
                    orders ON c_custkey = o_custkey
                JOIN
                    lineitem ON o_orderkey = l_orderkey
                WHERE
                    l_discount BETWEEN {min_discount} AND {max_discount}
                GROUP BY
                    c_name,
                    o_orderstatus
                ORDER BY
                    revenue DESC;
            """,
            8: """
                SELECT
                    s_name,
                    ps_partkey,
                    MAX(ps_availqty) AS max_qty
                FROM
                    supplier
                JOIN
                    partsupp ON s_suppkey = ps_suppkey
                WHERE
                    ps_supplycost < {cost_threshold}
                GROUP BY
                    s_name,
                    ps_partkey
                ORDER BY
                    max_qty DESC;
            """,
            9: """
                SELECT
                    l_orderkey,
                    l_returnflag,
                    COUNT(DISTINCT l_partkey) AS unique_parts
                FROM
                    lineitem
                WHERE
                    l_receiptdate > l_commitdate
                    AND l_quantity BETWEEN {min_quantity} AND {max_quantity}
                    AND l_returnflag = '{returnflag}'
                    AND l_shipdate BETWEEN DATE '{start_shipdate}' AND DATE '{end_shipdate}'
                GROUP BY
                    l_orderkey,
                    l_returnflag
                ORDER BY
                    unique_parts DESC;
            """,
            10: """
                SELECT
                    r_name,
                    SUM(o_totalprice) AS total_sales
                FROM
                    region
                JOIN
                    nation ON r_regionkey = n_regionkey
                JOIN
                    customer ON n_nationkey = c_nationkey
                JOIN
                    orders ON c_custkey = o_custkey
                WHERE
                    o_orderdate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                GROUP BY
                    r_name
                ORDER BY
                    total_sales DESC;
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
                    ps_partkey,
                    COUNT(DISTINCT ps_suppkey) AS num_suppliers
                FROM
                    partsupp
                WHERE
                    ps_supplycost BETWEEN {min_cost} AND {max_cost}
                GROUP BY
                    ps_partkey
                ORDER BY
                    num_suppliers DESC;
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
                    COUNT(l.l_orderkey) AS order_count
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
                    AND o.o_orderdate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                GROUP BY
                    c.c_custkey,
                    c.c_name,
                    o.o_orderdate,
                    s.s_name
                ORDER BY
                    order_count DESC;
            """,
        }

        self.predicates = {
            1: {
                'customer': ["c_region"],
                'lineitem': ["l_shipdate"]
            },
            2: {
                'supplier': ["s_acctbal"],
                'part': ["p_size"]
            },
            3: {
                'customer': ["c_acctbal"]
            },
            4: {
                'region': ["r_name"]
            },
            5: {
                'orders': ["o_totalprice", "o_orderstatus", "o_orderpriority", "o_orderdate"]
            },
            6: {
                'part': ["p_retailprice", "p_brand"],
                'lineitem': ["l_shipdate"]
            },
            7: {
                'lineitem': ["l_discount"]
            },
            8: {
                'partsupp': ["ps_supplycost"]
            },
            9: {
            'lineitem': ["l_receiptdate", "l_quantity", "l_returnflag", "l_shipdate"]
            },
            10: {
                'orders': ["o_orderdate"]
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
                "customer": ["c_custkey"],
                "orders": ["o_orderdate"],
                "lineitem": ["l_extendedprice", "l_discount"]
            },
            2: {
                "supplier": ["s_suppkey", "s_name"],
                "part": ["p_type"],
                "partsupp": ["ps_supplycost"]
            },
            3: {
                "nation": ["n_name"],
                "customer": ["c_custkey"]
            },
            4: {
                "region": ["r_name"],
                "part": ["p_mfgr"],
                "lineitem": ["l_quantity"]
            },
            5: {
                "orders": ["o_orderpriority"]
            },
            6: {
                "part": ["p_partkey", "p_name"],
                "lineitem": ["l_shipmode"]
            },
            7: {
                "customer": ["c_name"],
                "orders": ["o_orderstatus"],
                "lineitem": ["l_extendedprice"]
            },
            8: {
                "supplier": ["s_name"],
                "partsupp": ["ps_partkey", "ps_availqty"]
            },
            9: {
                "lineitem": ["l_partkey"]
            },
            10: {
                "region": ["r_name"],
                "orders": ["o_totalprice"]
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
                "partsupp": ["ps_partkey", "ps_suppkey"]
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
            1: {
                "": ["total_spent"]
            },
            2: {
                "": ["total_cost"]
            },
            3: {
                "": ["num_customers"]
            },
            4: {
                "": ["avg_quantity"]
            },
            5: {
                "": ["order_count"]
            },
            6: {
                "": ["ship_count"]
            },
            7: {
                "": ["revenue"]
            },
            8: {
                "": ["max_qty"]
            },
            9: {
                "lineitem": ["l_orderkey", "l_returnflag"]
            },
            10: {
                "": ["total_sales"]
            },
            11: {
                "": ["total_inventory_cost"]
            },
            12: {
                "": ["ps_availqty"]
            },
            13: {
                "": ["num_suppliers"]
            },
            14: {
                "": ["total_quantity"]
            },
            15: {
                "": ["avg_cost"]
            },
            16: {
                "": ["order_count"]
            }
        }

        self.group_bys = {
            1: {
                "customer": ["c_custkey"],
                "orders": ["o_orderdate"]
            },
            2: {
                "supplier": ["s_suppkey", "s_name"],
                "part": ["p_type"]
            },
            3: {
                "nation": ["n_name"]
            },
            4: {
                "region": ["r_name"],
                "part": ["p_mfgr"]
            },
            5: {
                "orders": ["o_orderpriority"]
            },
            6: {
                "part": ["p_partkey", "p_name"],
                "lineitem": ["l_shipmode"]
            },
            7: {
                "customer": ["c_name"],
                "orders": ["o_orderstatus"]
            },
            8: {
                "supplier": ["s_name"],
                "partsupp": ["ps_partkey"]
            },
            9: {
                "": ["unique_parts"]
            },
            10: {
                "region": ["r_name"]
            },
            11: {
                "partsupp": ["ps_partkey", "ps_suppkey"]
            },
            12: {},
            13: {
                "partsupp": ["ps_partkey"]
            },
            14: {
                "supplier": ["s_name"]
            },
            15: {
                "partsupp": ["ps_partkey"],
                "part": ["p_type"]
            },
            16: {
                "customer": ["c_custkey", "c_name"],
                "orders": ["o_orderdate"],
                "supplier": ["s_name"]
            }
        }


        self.predicate_dicts = {
            1: {
                "customer": [
                    {"column": "c_region", "operator": "eq", "value": "'{region}'", "join": False},
                    {"column": "c_custkey", "operator": "eq", "value": "o_custkey", "join": True},
                ],
                "lineitem": [
                    {"column": "l_shipdate", "operator": "range", "value": "'{start_date}' AND DATE '{end_date}'", "join": False},
                    {"column": "l_orderkey", "operator": "eq", "value": "o_orderkey", "join": True}
                ],
            },
            2: {
                "supplier": [
                    {"column": "s_acctbal", "operator": ">", "value": "{acctbal_threshold}", "join": False},
                    {"column": "s_suppkey", "operator": "eq", "value": "ps_suppkey", "join": True}
                ],
                "part": [
                    {"column": "p_size", "operator": "range", "value": "{min_size} AND {max_size}", "join": False},
                    {"column": "p_partkey", "operator": "eq", "value": "ps_partkey", "join": True}
                ],
            },
            3: {
                "customer": [
                    {"column": "c_acctbal", "operator": "<", "value": "{acctbal_threshold}", "join": False},
                    {"column": "c_nationkey", "operator": "eq", "value": "n_nationkey", "join": True}
                ]
            },
            4: {
                "region": [
                    {"column": "r_name", "operator": "like", "value": "'{region_pattern}'", "join": False},
                    {"column": "r_regionkey", "operator": "eq", "value": "n_regionkey", "join": True}
                ],
                "supplier": [
                    {"column": "s_nationkey", "operator": "eq", "value": "n_nationkey", "join": True}
                ],
                "lineitem": [
                    {"column": "l_suppkey", "operator": "eq", "value": "s_suppkey", "join": True}
                ],
                "part": [
                    {"column": "p_partkey", "operator": "eq", "value": "l_partkey", "join": True}
                ]
            },
            5: {
                "orders": [
                    {"column": "o_totalprice", "operator": ">", "value": "{price_threshold}", "join": False},
                    {"column": "o_orderstatus", "operator": "eq", "value": "'{status}'", "join": False},
                    {"column": "o_orderdate", "operator": "range", "value": "DATE '{start_date}' AND DATE '{end_date}'", "join": False},
                    {"column": "o_orderpriority", "operator": "in", "value": "('1-URGENT', '2-HIGH')", "join": False}
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
                "lineitem": [
                    {"column": "l_discount", "operator": "range", "value": "{min_discount} AND {max_discount}", "join": False},
                    {"column": "l_orderkey", "operator": "eq", "value": "o_orderkey", "join": True}
                ],
                "customer": [
                    {"column": "c_custkey", "operator": "eq", "value": "o_custkey", "join": True},
                ]
            },
            8: {
                "partsupp": [
                    {"column": "ps_supplycost", "operator": "<", "value": "{cost_threshold}", "join": False},
                    {"column": "ps_suppkey", "operator": "eq", "value": "s_suppkey", "join": True}
                ]
            },
            9: {
                "lineitem": [
                    {"column": "l_receiptdate", "operator": ">", "value": "l_commitdate", "join": False},
                    {"column": "l_quantity", "operator": "range", "value": "{min_quantity} AND {max_quantity}", "join": False},
                    {"column": "l_returnflag", "operator": "eq", "value": "'{returnflag}'", "join": False},
                    {"column": "l_shipdate", "operator": "range", "value": "DATE '{start_shipdate}' AND DATE '{end_shipdate}'", "join": False}
                ]
            },
            10: {
                "orders": [
                    {"column": "o_orderdate", "operator": "range", "value": "DATE '{start_date}' AND DATE '{end_date}'", "join": False},
                    {"column": "o_custkey", "operator": "eq", "value": "c_custkey", "join": True}
                ],
                "customer": [
                    {"column": "c_nationkey", "operator": "eq", "value": "n_nationkey", "join": True}
                ],
                "nation": [
                    {"column": "n_regionkey", "operator": "eq", "value": "r_regionkey", "join": True}
                ],
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
                    {"column": "ps_suppkey", "operator": "eq", "value": "s_suppkey", "join": True}
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
            region = random.choice(list(self.stats['region']['r_name']['histogram'].keys()))
            start_date = self.stats['lineitem']['l_shipdate']['min'] + timedelta(days=random.randint(0, (self.stats['lineitem']['l_shipdate']['max'] - self.stats['lineitem']['l_shipdate']['min']).days))
            end_date = start_date + timedelta(days=random.randint(1, 30))
            query = template.format(region=region, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            self.predicate_dicts[template_num]['customer'][0]['value'] = region
            self.predicate_dicts[template_num]['lineitem'][0]['value'] = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        elif template_num == 2:
            # Choose a realistic account balance threshold within supplier's account balance range
            acctbal_threshold = random.uniform(float(self.stats['supplier']['s_acctbal']['min']), float(self.stats['supplier']['s_acctbal']['max']))
            
            # Choose a realistic size range for parts
            min_size = random.randint(self.stats['part']['p_size']['min'], self.stats['part']['p_size']['max'] - 1)
            max_size = random.randint(min_size + 1, self.stats['part']['p_size']['max'])

            query = template.format(acctbal_threshold=acctbal_threshold, min_size=min_size, max_size=max_size)
            self.predicate_dicts[template_num]['supplier'][0]['value'] = acctbal_threshold
            self.predicate_dicts[template_num]['part'][0]['value'] = (min_size, max_size)

        elif template_num == 3:
            acctbal_threshold = random.uniform(float(self.stats['customer']['c_acctbal']['min']), float(self.stats['customer']['c_acctbal']['max']))
            query = template.format(acctbal_threshold=acctbal_threshold)
            self.predicate_dicts[template_num]['customer'][0]['value'] = acctbal_threshold

        elif template_num == 4:
            region_pattern = random.choice(list(self.stats['region']['r_name']['histogram'].keys()))
            query = template.format(region_pattern=region_pattern)
            self.predicate_dicts[template_num]['region'][0]['value'] = region_pattern

        elif template_num == 5:
            price_threshold = random.uniform(float(self.stats['orders']['o_totalprice']['min']), float(self.stats['orders']['o_totalprice']['max']))
            status = random.choice(list(self.stats['orders']['o_orderstatus']['histogram'].keys()))
            start_date = self.stats['orders']['o_orderdate']['min'] + timedelta(days=random.randint(0, (self.stats['orders']['o_orderdate']['max'] - self.stats['orders']['o_orderdate']['min']).days))
            end_date = start_date + timedelta(days=random.randint(1, 30))
            query = template.format(price_threshold=price_threshold, status=status, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            self.predicate_dicts[template_num]['orders'][0]['value'] = price_threshold
            self.predicate_dicts[template_num]['orders'][1]['value'] = status
            self.predicate_dicts[template_num]['orders'][2]['value'] =  (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        elif template_num == 6:
            min_price = float(self.stats['part']['p_retailprice']['min'])
            max_price = float(self.stats['part']['p_retailprice']['max'])
            random_min_price = random.uniform(min_price, max_price - 500)
            random_max_price = random.uniform(random_min_price + 1, max_price)
            brand = random.choice(list(self.stats['part']['p_brand']['histogram'].keys()))
            start_date = self.stats['lineitem']['l_shipdate']['min'] + timedelta(days=random.randint(0, (self.stats['lineitem']['l_shipdate']['max'] - self.stats['lineitem']['l_shipdate']['min']).days))
            end_date = start_date + timedelta(days=random.randint(1, 30))
            query = template.format(min_price=random_min_price, max_price=random_max_price, brand=brand, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            self.predicate_dicts[template_num]['part'][0]['value'] = (random_min_price, random_max_price)
            self.predicate_dicts[template_num]['part'][1]['value'] = brand
            self.predicate_dicts[template_num]['lineitem'][1]['value'] = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        elif template_num == 7:
            min_discount = float(self.stats['lineitem']['l_discount']['min'])
            max_discount = float(self.stats['lineitem']['l_discount']['max'])
            random_min_discount = random.uniform(min_discount, max_discount - 0.05)
            random_max_discount = random.uniform(random_min_discount + 0.01, max_discount)
            query = template.format(min_discount=random_min_discount, max_discount=random_max_discount)
            self.predicate_dicts[template_num]['lineitem'][0]['value'] = (random_min_discount, random_max_discount)

        elif template_num == 8:
            cost_threshold = random.uniform(float(self.stats['partsupp']['ps_supplycost']['min']), float(self.stats['partsupp']['ps_supplycost']['max']))
            query = template.format(cost_threshold=cost_threshold)
            self.predicate_dicts[template_num]['partsupp'][0]['value'] = cost_threshold

        elif template_num == 9:
            # Select a realistic quantity range
            min_quantity = float(self.stats['lineitem']['l_quantity']['min'])
            max_quantity = float(self.stats['lineitem']['l_quantity']['max'])
            random_min_quantity = random.uniform(min_quantity, max_quantity - 10)
            random_max_quantity = random.uniform(random_min_quantity + 1, max_quantity)

            # Select a random return flag from the distinct values
            returnflag = random.choice(list(self.stats['lineitem']['l_returnflag']['histogram'].keys()))

            # Select a realistic shipdate range
            shipdate_min = self.stats['lineitem']['l_shipdate']['min']
            shipdate_max = self.stats['lineitem']['l_shipdate']['max']
            start_shipdate = shipdate_min + timedelta(days=random.randint(0, (shipdate_max - shipdate_min).days - 30))
            end_shipdate = start_shipdate + timedelta(days=random.randint(1, 30))

            # Format the query with the selected parameters
            query = template.format(
                min_quantity=random_min_quantity,
                max_quantity=random_max_quantity,
                returnflag=returnflag,
                start_shipdate=start_shipdate.strftime('%Y-%m-%d'),
                end_shipdate=end_shipdate.strftime('%Y-%m-%d')
            )

            # Update the predicate dictionary
            self.predicate_dicts[9]['lineitem'][1]['value'] = (random_min_quantity, random_max_quantity)
            self.predicate_dicts[9]['lineitem'][2]['value'] = returnflag
            self.predicate_dicts[9]['lineitem'][3]['value'] = (start_shipdate.strftime('%Y-%m-%d'), end_shipdate.strftime('%Y-%m-%d'))

        elif template_num == 10:
            start_date = self.stats['orders']['o_orderdate']['min'] + timedelta(days=random.randint(0, (self.stats['orders']['o_orderdate']['max'] - self.stats['orders']['o_orderdate']['min']).days))
            end_date = start_date + timedelta(days=random.randint(1, 30))
            query = template.format(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            self.predicate_dicts[template_num]['orders'][0]['value'] = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

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
























