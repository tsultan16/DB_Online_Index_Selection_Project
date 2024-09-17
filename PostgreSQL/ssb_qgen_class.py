"""
    SSB query generator class
"""
import pickle
import random
import math


# returns SSB schema as a dictionary
def get_ssb_schema():
    tables = {
        "lineorder": [
            ("lo_orderkey", "INT"),
            ("lo_linenumber", "INT"),
            ("lo_custkey", "INT"),
            ("lo_partkey", "INT"),
            ("lo_suppkey", "INT"),
            ("lo_orderdate", "DATE"),
            ("lo_orderpriority", "CHAR(15)"),
            ("lo_shippriority", "CHAR(1)"),
            ("lo_quantity", "INT"),
            ("lo_extendedprice", "DECIMAL(18,2)"),
            ("lo_ordtotalprice", "DECIMAL(18,2)"),
            ("lo_discount", "DECIMAL(18,2)"),
            ("lo_revenue", "DECIMAL(18,2)"),
            ("lo_supplycost", "DECIMAL(18,2)"),
            ("lo_tax", "INT"),
            ("lo_commitdate", "DATE"),
            ("lo_shipmode", "CHAR(10)")
        ],
        "part": [
            ("p_partkey", "INT"),
            ("p_name", "VARCHAR(22)"),
            ("p_mfgr", "CHAR(6)"),
            ("p_category", "CHAR(7)"),
            ("p_brand", "CHAR(9)"),
            ("p_color", "VARCHAR(11)"),
            ("p_type", "VARCHAR(25)"),
            ("p_size", "INT"),
            ("p_container", "CHAR(15)")
        ],
        "supplier": [
            ("s_suppkey", "INT"),
            ("s_name", "CHAR(25)"),
            ("s_address", "VARCHAR(25)"),
            ("s_city", "CHAR(10)"),
            ("s_nation", "CHAR(15)"),
            ("s_region", "CHAR(12)"),
            ("s_phone", "CHAR(20)")
        ],
        "customer": [
            ("c_custkey", "INT"),
            ("c_name", "VARCHAR(25)"),
            ("c_address", "VARCHAR(25)"),
            ("c_city", "CHAR(10)"),
            ("c_nation", "CHAR(15)"),
            ("c_region", "CHAR(12)"),
            ("c_phone", "CHAR(15)"),
            ("c_mktsegment", "CHAR(12)")
        ],
        "dwdate": [
            ("d_datekey", "DATE"),
            ("d_date", "CHAR(18)"),
            ("d_dayofweek", "CHAR(9)"),
            ("d_month", "CHAR(9)"),
            ("d_year", "INT"),
            ("d_yearmonthnum", "INT"),
            ("d_yearmonth", "CHAR(7)"),
            ("d_daynuminweek", "INT"),
            ("d_daynuminmonth", "INT"),
            ("d_daynuminyear", "INT"),
            ("d_monthnuminyear", "INT"),
            ("d_weeknuminyear", "INT"),
            ("d_sellingseason", "CHAR(12)"),
            ("d_lastdayinweekfl", "BIT"),
            ("d_lastdayinmonthfl", "BIT"),
            ("d_holidayfl", "BIT"),
            ("d_weekdayfl", "BIT")
        ]
    }

    pk_columns = {"lineorder": ["lo_orderkey", "lo_linenumber"], "part": ["p_partkey"], "supplier": ["s_suppkey"], "customer": ["c_custkey"], "dwdate": ["d_datekey"]}

    return tables, pk_columns


# class for SSB queries
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


# class for generating SSB queries
class QGEN:

    def __init__(self, seed=1234):
        if seed is not None:
            random.seed(seed)

        # assume table self.stats and schema available
        with open('ssb_stats.pkl', 'rb') as f:
            self.stats = pickle.load(f)

        self.query_templates = {    
            1: """
                SELECT SUM(lo_extendedprice * lo_discount) AS revenue
                FROM lineorder, dwdate
                WHERE lo_orderdate = d_datekey
                AND d_year = {year}
                AND lo_discount BETWEEN {discount_low} AND {discount_high} 
                AND lo_quantity {inequality} {quantity};
            """,
            2: """
                SELECT SUM(lo_extendedprice * lo_discount) AS revenue
                FROM lineorder, dwdate
                WHERE lo_orderdate = d_datekey
                AND d_yearmonthnum = {yearmonthnum}
                AND lo_discount BETWEEN {discount_low}  AND {discount_high} 
                AND lo_quantity BETWEEN {quantity_low} AND {quantity_high};
            """,
            3: """
                SELECT SUM(lo_extendedprice * lo_discount) AS revenue
                FROM lineorder, dwdate
                WHERE lo_orderdate = d_datekey
                AND d_weeknuminyear = {weeknuminyear}
                AND d_year = {year}
                AND lo_discount BETWEEN {discount_low} AND {discount_high}
                AND lo_quantity BETWEEN {quantity_low} AND {quantity_high};
            """,

            4: """
                SELECT SUM(lo_revenue), d_year, p_brand
                FROM lineorder, dwdate, part, supplier
                WHERE lo_orderdate = d_datekey
                AND lo_partkey = p_partkey
                AND lo_suppkey = s_suppkey
                AND p_category = '{category}'
                AND s_region = '{sregion}'
                GROUP BY d_year, p_brand
                ORDER BY d_year, p_brand;
            """,
            5: """
                SELECT SUM(lo_revenue), d_year, p_brand
                FROM lineorder, dwdate, part, supplier
                WHERE lo_orderdate = d_datekey
                AND lo_partkey = p_partkey
                AND lo_suppkey = s_suppkey
                AND p_brand BETWEEN '{brand1_low}' AND '{brand1_high}'
                AND s_region = '{sregion}'
                GROUP BY d_year, p_brand
                ORDER BY d_year, p_brand;
            """,
            6: """
                SELECT SUM(lo_revenue), d_year, p_brand
                FROM lineorder, dwdate, part, supplier
                WHERE lo_orderdate = d_datekey
                AND lo_partkey = p_partkey
                AND lo_suppkey = s_suppkey
                AND p_brand = '{brand1}'
                AND s_region = '{sregion}'
                GROUP BY d_year, p_brand
                ORDER BY d_year, p_brand;
            """,
                        
            7: """
                SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue
                FROM customer, lineorder, supplier, dwdate
                WHERE lo_custkey = c_custkey
                AND lo_suppkey = s_suppkey
                AND lo_orderdate = d_datekey
                AND c_region = '{region}'
                AND s_region = '{region}'
                AND d_year >= {year_low} AND d_year <= {year_high}
                GROUP BY c_nation, s_nation, d_year
                ORDER BY d_year ASC, revenue DESC;
            """,
            8: """
                SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
                FROM customer, lineorder, supplier, dwdate
                WHERE lo_custkey = c_custkey
                AND lo_suppkey = s_suppkey
                AND lo_orderdate = d_datekey
                AND c_nation = '{region}'
                AND s_nation = '{region}'
                AND d_year >= {year_low} AND d_year <= {year_high}
                GROUP BY c_city, s_city, d_year
                ORDER BY d_year ASC, revenue DESC;
            """,
            9: """
                SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
                FROM customer, lineorder, supplier, dwdate
                WHERE lo_custkey = c_custkey
                AND lo_suppkey = s_suppkey
                AND lo_orderdate = d_datekey
                AND (c_city = '{city_1}' OR c_city = '{city_2}')
                AND (s_city = '{city_1}' OR s_city = '{city_2}')
                AND d_year >= {year_low} AND d_year <= {year_high}
                GROUP BY c_city, s_city, d_year
                ORDER BY d_year ASC, revenue DESC;
            """,
            10: """
                SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
                FROM customer, lineorder, supplier, dwdate
                WHERE lo_custkey = c_custkey
                AND lo_suppkey = s_suppkey
                AND lo_orderdate = d_datekey
                AND (c_city = '{city_1}' OR c_city = '{city_2}')
                AND (s_city = '{city_1}' OR s_city = '{city_2}')
                AND d_yearmonth = '{yearmonth}'
                GROUP BY c_city, s_city, d_year
                ORDER BY d_year ASC, revenue DESC;
            """ ,

            11: """
                SELECT d_year, c_nation, SUM(lo_revenue - lo_supplycost) AS profit
                FROM dwdate, customer, supplier, part, lineorder
                WHERE lo_custkey = c_custkey
                AND lo_suppkey = s_suppkey
                AND lo_partkey = p_partkey
                AND lo_orderdate = d_datekey
                AND c_region = '{region}'
                AND s_region = '{region}'
                AND (p_mfgr = '{mfgr_1}' OR p_mfgr = '{mfgr_2}')
                GROUP BY d_year, c_nation
                ORDER BY d_year, c_nation;
            """,
            12: """
                SELECT d_year, s_nation, p_category, SUM(lo_revenue - lo_supplycost) AS profit
                FROM dwdate, customer, supplier, part, lineorder
                WHERE lo_custkey = c_custkey
                AND lo_suppkey = s_suppkey
                AND lo_partkey = p_partkey
                AND lo_orderdate = d_datekey
                AND c_region = '{region}'
                AND s_region = '{region}'
                AND (d_year = {year_1} OR d_year = {year_2})
                AND (p_mfgr = '{mfgr_1}' OR p_mfgr = '{mfgr_2}')
                GROUP BY d_year, s_nation, p_category
                ORDER BY d_year, s_nation, p_category;
            """,
            13: """
                SELECT d_year, s_city, p_brand, SUM(lo_revenue - lo_supplycost) AS profit
                FROM dwdate, customer, supplier, part, lineorder
                WHERE lo_custkey = c_custkey
                AND lo_suppkey = s_suppkey
                AND lo_partkey = p_partkey
                AND lo_orderdate = d_datekey
                AND c_region = '{region}'
                AND s_nation = '{nation}'
                AND (d_year = {year_1} OR d_year = {year_2})
                AND p_category = '{category}'
                GROUP BY d_year, s_city, p_brand
                ORDER BY d_year, s_city, p_brand;
            """,
            
            14: """
                SELECT lo_linenumber, lo_quantity  
                FROM lineorder
                WHERE lo_linenumber >= {linenumber_low} AND lo_linenumber <= {linenumber_high}
                AND lo_quantity = {quantity};
            """, 

            15: """
                SELECT lo_linenumber, lo_quantity, lo_orderdate  
                FROM lineorder
                WHERE lo_linenumber >= {linenumber_low} AND lo_linenumber <= {linenumber_high}
                AND lo_quantity = {quantity};
            """, 
            16: """
                SELECT lo_linenumber, lo_extendedprice  
                FROM lineorder
                WHERE lo_extendedprice = {extendedprice};
            """,
            }
        
        self.predicates = {1: {"lineorder": ["lo_orderdate", "lo_discount", "lo_quantity"], "dwdate": ["d_datekey", "d_year"]},
              2: {"lineorder": ["lo_orderdate", "lo_discount", "lo_quantity"],
                  "dwdate": ["d_datekey", "d_yearmonthnum"]},
              3: {"lineorder": ["lo_orderdate", "lo_discount", "lo_quantity"],
                  "dwdate": ["d_datekey", "d_weeknuminyear"]},
              4: {"lineorder": ["lo_orderdate", "lo_partkey", "lo_suppkey"], "dwdate": ["d_year", "d_datekey"],
                  "part": ["p_partkey", "p_category", "p_brand"], "supplier": ["s_suppkey", "s_region"]},
              5: {"lineorder": ["lo_orderdate", "lo_partkey", "lo_suppkey"], "dwdate": ["d_year", "d_datekey"],
                  "part": ["p_partkey", "p_category", "p_brand"], "supplier": ["s_suppkey", "s_region"]},
              6: {"lineorder": ["lo_orderdate", "lo_partkey", "lo_suppkey"], "dwdate": ["d_year", "d_datekey"],
                  "part": ["p_partkey", "p_category", "p_brand"], "supplier": ["s_suppkey", "s_region"]},
              7: {"lineorder": ["lo_custkey", "lo_suppkey", "lo_orderdate"], "dwdate": ["d_year", "d_datekey"],
                  "customer": ["c_custkey", "c_region", "c_nation"], "supplier": ["s_suppkey", "s_region", "s_nation"]},
              8: {"lineorder": ["lo_custkey", "lo_suppkey", "lo_orderdate"], "dwdate": ["d_year", "d_datekey"],
                  "customer": ["c_custkey", "c_nation", "c_city"], "supplier": ["s_suppkey", "s_nation", "s_city"]},
              9: {"lineorder": ["lo_custkey", "lo_suppkey", "lo_orderdate"], "dwdate": ["d_year", "d_datekey"],
                  "customer": ["c_custkey", "c_city"], "supplier": ["s_suppkey", "s_city"]},
              10: {"lineorder": ["lo_custkey", "lo_suppkey", "lo_orderdate"], "dwdate": ["d_yearmonth", "d_datekey"],
                   "customer": ["c_custkey", "c_city"], "supplier": ["s_suppkey", "s_city"]},
              11: {"lineorder": ["lo_custkey", "lo_suppkey", "lo_orderdate", "lo_partkey"],
                   "dwdate": ["d_year", "d_datekey"], "customer": ["c_custkey", "c_region", "c_nation"],
                   "part": ["p_partkey", "p_mfgr"], "supplier": ["s_suppkey", "s_region"]},
              12: {"lineorder": ["lo_custkey", "lo_suppkey", "lo_orderdate", "lo_partkey"],
                   "dwdate": ["d_year", "d_datekey"], "customer": ["c_custkey", "c_region", "c_nation"],
                   "part": ["p_partkey", "p_mfgr", "p_category"], "supplier": ["s_suppkey", "s_region"]},
              13: {"lineorder": ["lo_custkey", "lo_suppkey", "lo_orderdate", "lo_partkey"],
                   "dwdate": ["d_year", "d_datekey"], "customer": ["c_custkey", "c_nation"],
                   "part": ["p_partkey", "p_mfgr", "p_category"], "supplier": ["s_suppkey", "s_nation"]},
              14: {"lineorder": ["lo_linenumber", "lo_quantity"]},  
              15: {"lineorder": ["lo_linenumber", "lo_quantity"]},  
              16: {"lineorder": ["lo_extendedprice"]},
              }

        self.payloads = {1: {"lineorder": ["lo_extendedprice", "lo_discount"]},
                    2: {"lineorder": ["lo_extendedprice", "lo_discount"]},
                    3: {"lineorder": ["lo_extendedprice", "lo_discount"]},
                    4: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"], "part": ["p_brand"]},
                    5: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"], "part": ["p_brand"]},
                    6: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"], "part": ["p_brand"]},
                    7: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"], "customer": ["c_nation"], "supplier": ["s_nation"]},
                    8: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"], "customer": ["c_city"], "supplier": ["s_city"]},
                    9: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"], "customer": ["c_city"], "supplier": ["s_city"]},
                    10: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"], "customer": ["c_city"], "supplier": ["s_city"]},
                    11: {"lineorder": ["lo_revenue", "lo_supplycost"], "dwdate": ["d_year"], "customer": ["c_nation"]},
                    12: {"lineorder": ["lo_revenue", "lo_supplycost"], "dwdate": ["d_year"], "part": ["p_category"],
                        "supplier": ["s_nation"]},
                    13: {"lineorder": ["lo_revenue", "lo_supplycost"], "dwdate": ["d_year"], "part": ["p_brand"],
                        "supplier": ["s_city"]},
                    14: {"lineorder": ["lo_linenumber", "lo_quantity"]},  
                    
                    15: {"lineorder": ["lo_linenumber", "lo_quantity", "lo_orderdate"]}, 
                    16: {"lineorder": ["lo_extendedprice", "lo_linenumber"]}, 
                    }

        self.order_bys = {1: {},
                    2: {},
                    3: {},
                    4: {"dwdate": ["d_year"], "part": ["p_brand"]},
                    5: {"dwdate": ["d_year"], "part": ["p_brand"]},
                    6: {"dwdate": ["d_year"], "part": ["p_brand"]},
                    7: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"]},
                    8: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"]},
                    9: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"]},
                    10: {"lineorder": ["lo_revenue"], "dwdate": ["d_year"]},
                    11: {"dwdate": ["d_year"], "customer": ["c_nation"]},
                    12: {"dwdate": ["d_year"], "part": ["p_category"], "supplier": ["s_nation"]},
                    13: {"dwdate": ["d_year"], "part": ["p_brand"], "supplier": ["s_city"]},
                    14: {},
                    15: {},
                    16: {},
                    }

        self.group_bys = {1: {},
                    2: {},
                    3: {},
                    4: {"dwdate": ["d_year"], "part": ["p_brand"]}, 
                    5: {"dwdate": ["d_year"], "part": ["p_brand"]},
                    6: {"dwdate": ["d_year"], "part": ["p_brand"]},
                    7: {"customer": ["c_nation"], "supplier": ["s_nation"], "dwdate": ["d_year"]},
                    8: {"customer": ["c_city"], "supplier": ["s_city"], "dwdate": ["d_year"]},
                    9: {"customer": ["c_city"], "supplier": ["s_city"], "dwdate": ["d_year"]},
                    10: {"customer": ["c_city"], "supplier": ["s_city"], "dwdate": ["d_year"]},
                    11: {"customer": ["c_nation"], "dwdate": ["d_year"]},
                    12: {"part": ["p_category"], "supplier": ["s_nation"], "dwdate": ["d_year"]},
                    13: {"part": ["p_brand"], "supplier": ["s_city"], "dwdate": ["d_year"]},
                    14: {},
                    15: {},
                    16: {},
                    }

        self.predicate_dicts = {1: {'lineorder': [
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True},
                                        {'column': 'lo_discount', 'operator': 'range', 'value': ('discount_low', 'discount_high'), 'join': False},
                                        {'column': 'lo_quantity', 'operator': 'inequality', 'value': 'quantity', 'join': False}],
                                    'dwdate': [
                                        {'column': 'd_year', 'operator': 'eq', 'value': 'year'}]}, 
                                
                                2: {'lineorder': [
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True},
                                        {'column': 'lo_discount', 'operator': 'range', 'value': ('discount_low', 'discount_high'), 'join': False},
                                        {'column': 'lo_quantity', 'operator': 'range', 'value': ('quantity_low', 'quantity_high'), 'join': False}],
                                    'dwdate': [
                                        {'column': 'd_yearmonthnum', 'operator': 'eq', 'value': 'yearmonthnum', 'join': False}]},
                                
                                3: {'lineorder': [
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True},
                                        {'column': 'lo_discount', 'operator': 'range', 'value': ('discount_low', 'discount_high'), 'join': False},
                                        {'column': 'lo_quantity', 'operator': 'range', 'value': ('quantity_low', 'quantity_high'), 'join': False}],
                                    'dwdate': [
                                        {'column': 'd_weeknuminyear', 'operator': 'eq', 'value': 'weeknuminyear', 'join': False},
                                        {'column': 'd_year', 'operator': 'eq', 'value': 'year', 'join': False}]}, 

                                4: {'lineorder': [
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True},
                                        {'column': 'lo_partkey', 'operator': 'eq', 'value': 'p_partkey', 'join': True},
                                        {'column': 'lo_suppkey', 'operator': 'eq', 'value': 's_suppkey', 'join': True}
                                    ],
                                    'part': [
                                        {'column': 'p_category', 'operator': 'eq', 'value': 'category', 'join': False}
                                    ],
                                    'supplier': [
                                        {'column': 's_region', 'operator': 'eq', 'value': 'sregion', 'join': False}
                                    ]},

                                5: {'lineorder': [
                                            {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True},
                                            {'column': 'lo_partkey', 'operator': 'eq', 'value': 'p_partkey', 'join': True},
                                            {'column': 'lo_suppkey', 'operator': 'eq', 'value': 's_suppkey', 'join': True}
                                    ],
                                    'part': [
                                        {'column': 'p_brand', 'operator': 'range', 'value': ('brand1_low', 'brand1_high'), 'join': False}
                                    ],
                                    'supplier': [
                                        {'column': 's_region', 'operator': 'eq', 'value': 'sregion', 'join': False}
                                    ]}, 

                                6: {'lineorder': [
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True},
                                        {'column': 'lo_partkey', 'operator': 'eq', 'value': 'p_partkey', 'join': True},
                                        {'column': 'lo_suppkey', 'operator': 'eq', 'value': 's_suppkey', 'join': True}
                                    ],
                                    'part': [
                                        {'column': 'p_brand', 'operator': 'eq', 'value': 'brand1', 'join': False}
                                    ],
                                    'supplier': [
                                        {'column': 's_region', 'operator': 'eq', 'value': 'sregion', 'join': False}
                                    ]},

                                7: {'customer': [
                                        {'column': 'c_region', 'operator': 'eq', 'value': 'region', 'join': False}
                                    ],
                                    'lineorder': [
                                        {'column': 'lo_custkey', 'operator': 'eq', 'value': 'c_custkey', 'join': True},
                                        {'column': 'lo_suppkey', 'operator': 'eq', 'value': 's_suppkey', 'join': True},
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True}
                                    ],
                                    'supplier': [
                                        {'column': 's_region', 'operator': 'eq', 'value': 'region', 'join': False}
                                    ],
                                    'dwdate': [
                                        {'column': 'd_year', 'operator': 'range', 'value': ('year_low', 'year_high'), 'join': False}
                                    ]},

                                8: {'customer': [
                                        {'column': 'c_nation', 'operator': 'eq', 'value': 'region', 'join': False}
                                    ],
                                    'lineorder': [
                                        {'column': 'lo_custkey', 'operator': 'eq', 'value': 'c_custkey', 'join': True},
                                        {'column': 'lo_suppkey', 'operator': 'eq', 'value': 's_suppkey', 'join': True},
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True}
                                    ],
                                    'supplier': [
                                        {'column': 's_nation', 'operator': 'eq', 'value': 'region', 'join': False}
                                    ],
                                    'dwdate': [
                                        {'column': 'd_year', 'operator': 'range', 'value': ('year_low', 'year_high'), 'join': False}
                                    ]},

                                9: {'customer': [
                                        {'column': 'c_city', 'operator': 'in', 'value': ['city_1', 'city_2'], 'join': False}
                                    ],
                                    'lineorder': [
                                        {'column': 'lo_custkey', 'operator': 'eq', 'value': 'c_custkey', 'join': True},
                                        {'column': 'lo_suppkey', 'operator': 'eq', 'value': 's_suppkey', 'join': True},
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True}
                                    ],
                                    'supplier': [
                                        {'column': 's_city', 'operator': 'in', 'value': ['city_1', 'city_2'], 'join': False}
                                    ],
                                    'dwdate': [
                                        {'column': 'd_year', 'operator': 'range', 'value': ('year_low', 'year_high'), 'join': False}
                                    ]},  

                                10: {'customer': [
                                        {'column': 'c_city', 'operator': 'in', 'value': ['city_1', 'city_2'], 'join': False}
                                    ],
                                    'lineorder': [
                                        {'column': 'lo_custkey', 'operator': 'eq', 'value': 'c_custkey', 'join': True},
                                        {'column': 'lo_suppkey', 'operator': 'eq', 'value': 's_suppkey', 'join': True},
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True}
                                    ],
                                    'supplier': [
                                        {'column': 's_city', 'operator': 'in', 'value': ['city_1', 'city_2'], 'join': False}
                                    ],
                                    'dwdate': [
                                        {'column': 'd_yearmonth', 'operator': 'eq', 'value': 'yearmonth', 'join': False}
                                    ]},  

                                11: {'customer': [
                                        {'column': 'c_region', 'operator': 'eq', 'value': 'region', 'join': False}
                                    ],
                                    'supplier': [
                                        {'column': 's_region', 'operator': 'eq', 'value': 'region', 'join': False}
                                    ],
                                    'part': [
                                        {'column': 'p_mfgr', 'operator': 'in', 'value': ['mfgr_1', 'mfgr_2'], 'join': False}
                                    ],
                                    'lineorder': [
                                        {'column': 'lo_custkey', 'operator': 'eq', 'value': 'c_custkey', 'join': True},
                                        {'column': 'lo_suppkey', 'operator': 'eq', 'value': 's_suppkey', 'join': True},
                                        {'column': 'lo_partkey', 'operator': 'eq', 'value': 'p_partkey', 'join': True},
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True}
                                    ]}, 

                                12: {'dwdate': [
                                        {'column': 'd_year', 'operator': 'in', 'value': ['year_1', 'year_2'], 'join': False}
                                    ],
                                    'customer': [
                                        {'column': 'c_region', 'operator': 'eq', 'value': 'region', 'join': False}
                                    ],
                                    'supplier': [
                                        {'column': 's_region', 'operator': 'eq', 'value': 'region', 'join': False}
                                    ],
                                    'part': [
                                        {'column': 'p_mfgr', 'operator': 'in', 'value': ['mfgr_1', 'mfgr_2'], 'join': False}
                                    ],
                                    'lineorder': [
                                        {'column': 'lo_custkey', 'operator': 'eq', 'value': 'c_custkey', 'join': True},
                                        {'column': 'lo_suppkey', 'operator': 'eq', 'value': 's_suppkey', 'join': True},
                                        {'column': 'lo_partkey', 'operator': 'eq', 'value': 'p_partkey', 'join': True},
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True}
                                    ]}, 

                                13: {'dwdate': [
                                        {'column': 'd_year', 'operator': 'in', 'value': ['year_1', 'year_2'], 'join': False}
                                    ],
                                    'customer': [
                                        {'column': 'c_region', 'operator': 'eq', 'value': 'region', 'join': False}
                                    ],
                                    'supplier': [
                                        {'column': 's_nation', 'operator': 'eq', 'value': 'nation', 'join': False}
                                    ],
                                    'part': [
                                        {'column': 'p_category', 'operator': 'eq', 'value': 'category', 'join': False}
                                    ],
                                    'lineorder': [
                                        {'column': 'lo_custkey', 'operator': 'eq', 'value': 'c_custkey', 'join': True},
                                        {'column': 'lo_suppkey', 'operator': 'eq', 'value': 's_suppkey', 'join': True},
                                        {'column': 'lo_partkey', 'operator': 'eq', 'value': 'p_partkey', 'join': True},
                                        {'column': 'lo_orderdate', 'operator': 'eq', 'value': 'd_datekey', 'join': True}
                                    ]},

                                14:{'lineorder': [
                                        {'column': 'lo_linenumber', 'operator': 'range', 'value': ('linenumber_low', 'linenumber_high'), 'join': False},
                                        {'column': 'lo_quantity', 'operator': 'eq', 'value': 'quantity', 'join': False}
                                    ]},

                                15: {'lineorder': [
                                        {'column': 'lo_linenumber', 'operator': 'range', 'value': ('linenumber_low', 'linenumber_high'), 'join': False},
                                        {'column': 'lo_quantity', 'operator': 'eq', 'value': 'quantity', 'join': False}
                                    ]},              

                                16: {'lineorder': [
                                        {'column': 'lo_extendedprice', 'operator': 'eq', 'value': 'extendedprice', 'join': False}
                                    ]}

                                }


    def generate_query(self, template_num):
        if template_num not in self.query_templates:
            raise ValueError("Template not found")

        template = self.query_templates[template_num]

        # Fill parameters based on statistics
        if template_num == 1:
            year_stats = self.stats['dwdate']['d_year']
            discount_stats = self.stats['lineorder']['lo_discount']
            quantity_stats = self.stats['lineorder']['lo_quantity']

            year = random.choice(list(year_stats['histogram'].keys()))
            discount_low = math.floor(random.uniform(float(discount_stats['min']), float(discount_stats['max'])-2))
            discount_high = discount_low + 2
            quantity = 25 #random.randint(quantity_stats['min'], quantity_stats['max'])
            inequality_op = random.choice(['<', '>'])

            query = template.format(year=year, discount_low=discount_low, discount_high=discount_high, inequality=inequality_op, quantity=quantity)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['lineorder'][1]['value'] = (discount_low, discount_high)
            predicate_dict['lineorder'][2]['value'] = quantity
            predicate_dict['dwdate'][0]['value'] = year

        elif template_num == 2:
            yearmonthnum_stats = self.stats['dwdate']['d_yearmonthnum']
            discount_stats = self.stats['lineorder']['lo_discount']
            quantity_stats = self.stats['lineorder']['lo_quantity']

            yearmonthnum = random.choice(list(yearmonthnum_stats['histogram'].keys()))
            discount_low = math.floor(random.uniform(float(discount_stats['min']), float(discount_stats['max'])-2))
            discount_high = discount_low + 2
            quantity_low = random.randint(quantity_stats['min'], quantity_stats['max']-9)
            quantity_high = quantity_low + 9

            query = template.format(yearmonthnum=yearmonthnum, discount_low=discount_low, discount_high=discount_high, quantity_low=quantity_low, quantity_high=quantity_high)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['lineorder'][1]['value'] = (discount_low, discount_high)
            predicate_dict['lineorder'][2]['value'] = (quantity_low, quantity_high)
            predicate_dict['dwdate'][0]['value'] = yearmonthnum

        elif template_num == 3:
            weeknuminyear_stats = self.stats['dwdate']['d_weeknuminyear']
            year_stats = self.stats['dwdate']['d_year']
            discount_stats = self.stats['lineorder']['lo_discount']
            quantity_stats = self.stats['lineorder']['lo_quantity']

            weeknuminyear = random.choice(list(weeknuminyear_stats['histogram'].keys()))
            year = random.choice(list(year_stats['histogram'].keys()))
            discount_low = math.floor(random.uniform(float(discount_stats['min']), float(discount_stats['max'])-2))
            discount_high = discount_low + 2
            quantity_low = random.randint(quantity_stats['min'], quantity_stats['max']-9)
            quantity_high = quantity_low + 9

            query = template.format(weeknuminyear=weeknuminyear, year=year, discount_low=discount_low, discount_high=discount_high, quantity_low=quantity_low, quantity_high=quantity_high)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['lineorder'][1]['value'] = (discount_low, discount_high)
            predicate_dict['lineorder'][2]['value'] = (quantity_low, quantity_high)
            predicate_dict['dwdate'][0]['value'] = weeknuminyear
            predicate_dict['dwdate'][1]['value'] = year

        elif template_num == 4:
            category_stats = self.stats['part']['p_category']
            sregion_stats = self.stats['supplier']['s_region']

            category = random.choice(list(category_stats['histogram'].keys()))
            sregion = random.choice(list(sregion_stats['histogram'].keys()))

            query = template.format(category=category, sregion=sregion)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['part'][0]['value'] = category   
            predicate_dict['supplier'][0]['value'] = sregion            

        elif template_num == 5:
            brand1_stats = self.stats['part']['p_brand']
            sregion_stats = self.stats['supplier']['s_region']

            brand1_values = list(brand1_stats['histogram'].keys())
            #brand1_low = random.choice(brand1_values)
            #brand1_high = random.choice([b for b in brand1_values if b >= brand1_low])
            min_value = min(int(value.split('#')[-1]) for value in brand1_values)
            max_value = max(int(value.split('#')[-1]) for value in brand1_values)
            brand1_low_num = random.choice([i for i in range(min_value, max_value-7+1)])
            brand1_low = 'MFGR#' + str(brand1_low_num)
            brand1_high_num = brand1_low_num + 7
            brand1_high = 'MFGR#' + str(brand1_high_num)

            sregion = random.choice(list(sregion_stats['histogram'].keys()))
            query = template.format(brand1_low=brand1_low, brand1_high=brand1_high, sregion=sregion)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['part'][0]['value'] = (brand1_low, brand1_high)
            predicate_dict['supplier'][0]['value'] = sregion

        elif template_num == 6:
            brand1_stats = self.stats['part']['p_brand']
            sregion_stats = self.stats['supplier']['s_region']

            brand1 = random.choice(list(brand1_stats['histogram'].keys()))
            sregion = random.choice(list(sregion_stats['histogram'].keys()))

            query = template.format(brand1=brand1, sregion=sregion)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['part'][0]['value'] = brand1
            predicate_dict['supplier'][0]['value'] = sregion

        elif template_num == 7:
            region_stats = self.stats['customer']['c_region']
            year_stats = self.stats['dwdate']['d_year']

            region = random.choice(list(region_stats['histogram'].keys()))
            year_low = random.randint(year_stats['min'], year_stats['max']-5)
            year_high = year_low + 5

            query = template.format(region=region, year_low=year_low, year_high=year_high)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['customer'][0]['value'] = region
            predicate_dict['supplier'][0]['value'] = region
            predicate_dict['dwdate'][0]['value'] = (year_low, year_high)

        elif template_num == 8:
            region_stats = self.stats['customer']['c_nation']
            year_stats = self.stats['dwdate']['d_year']

            region = random.choice(list(region_stats['histogram'].keys()))
            year_low = random.randint(year_stats['min'], year_stats['max']-5)
            year_high = year_low + 5

            query = template.format(region=region, year_low=year_low, year_high=year_high)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['customer'][0]['value'] = region
            predicate_dict['supplier'][0]['value'] = region
            predicate_dict['dwdate'][0]['value'] = (year_low, year_high)

        elif template_num == 9:
            city_stats = self.stats['customer']['c_city']
            year_stats = self.stats['dwdate']['d_year']

            city_1 = random.choice(list(city_stats['histogram'].keys()))
            city_2 = random.choice([c for c in city_stats['histogram'].keys() if c != city_1])
            year_low = random.randint(year_stats['min'], year_stats['max']-5)
            year_high = year_low + 5

            query = template.format(city_1=city_1, city_2=city_2, year_low=year_low, year_high=year_high)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['customer'][0]['value'] = [city_1, city_2]
            predicate_dict['supplier'][0]['value'] = [city_1, city_2]
            predicate_dict['dwdate'][0]['value'] = (year_low, year_high)

        elif template_num == 10:
            city_stats = self.stats['customer']['c_city']
            yearmonth_stats = self.stats['dwdate']['d_yearmonth']

            city_1 = random.choice(list(city_stats['histogram'].keys()))
            city_2 = random.choice([c for c in city_stats['histogram'].keys() if c != city_1])
            yearmonth = random.choice(list(yearmonth_stats['histogram'].keys()))

            query = template.format(city_1=city_1, city_2=city_2, yearmonth=yearmonth)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['customer'][0]['value'] = [city_1, city_2]
            predicate_dict['supplier'][0]['value'] = [city_1, city_2]
            predicate_dict['dwdate'][0]['value'] = yearmonth

        elif template_num == 11:
            region_stats = self.stats['customer']['c_region']
            mfgr_stats = self.stats['part']['p_mfgr']

            region = random.choice(list(region_stats['histogram'].keys()))
            mfgr_values = list(mfgr_stats['histogram'].keys())
            min_value = min(int(value.split('#')[-1]) for value in mfgr_values)
            max_value = max(int(value.split('#')[-1]) for value in mfgr_values)
            mfgr_1_low_num = random.choice([i for i in range(min_value, max_value)])
            mfgr_1_low = 'MFGR#' + str(mfgr_1_low_num)
            mfgr_1_high_num = min(mfgr_1_low_num + random.randint(1, 3), max_value)
            mfgr_1_high = 'MFGR#' + str(mfgr_1_high_num)

            query = template.format(region=region, mfgr_1=mfgr_1_low, mfgr_2=mfgr_1_high)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['customer'][0]['value'] = region
            predicate_dict['part'][0]['value'] = (mfgr_1_low, mfgr_1_high)
            predicate_dict['supplier'][0]['value'] = region

        elif template_num == 12:
            region_stats = self.stats['customer']['c_region']
            year_stats = self.stats['dwdate']['d_year']
            mfgr_stats = self.stats['part']['p_mfgr']

            region = random.choice(list(region_stats['histogram'].keys()))
            year_low = random.randint(year_stats['min'], year_stats['max']-1)
            year_high = min(year_low + random.randint(1, 3), year_stats['max'])        
            mfgr_values = list(mfgr_stats['histogram'].keys())
            min_value = min(int(value.split('#')[-1]) for value in mfgr_values)
            max_value = max(int(value.split('#')[-1]) for value in mfgr_values)
            mfgr_1_low_num = random.choice([i for i in range(min_value, max_value)])
            mfgr_1_low = 'MFGR#' + str(mfgr_1_low_num)
            mfgr_1_high_num = min(mfgr_1_low_num + random.randint(1, 3), max_value)
            mfgr_1_high = 'MFGR#' + str(mfgr_1_high_num)

            query = template.format(region=region, year_1=year_low, year_2=year_high, mfgr_1=mfgr_1_low, mfgr_2=mfgr_1_high)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['customer'][0]['value'] = region
            predicate_dict['part'][0]['value'] = [mfgr_1_low, mfgr_1_high]
            predicate_dict['supplier'][0]['value'] = region
            predicate_dict['dwdate'][0]['value'] = [year_low, year_high]

        elif template_num == 13:
            region_stats = self.stats['customer']['c_region']
            nation_stats = self.stats['supplier']['s_nation']
            year_stats = self.stats['dwdate']['d_year']
            category_stats = self.stats['part']['p_category']

            region = random.choice(list(region_stats['histogram'].keys()))
            nation = random.choice(list(nation_stats['histogram'].keys()))
            year_low = random.randint(year_stats['min'], year_stats['max']-1)
            year_high = min(year_low + random.randint(1, 3), year_stats['max'])   
            category = random.choice(list(category_stats['histogram'].keys()))

            query = template.format(region=region, nation=nation, year_1=year_low, year_2=year_high, category=category)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['customer'][0]['value'] = region
            predicate_dict['part'][0]['value'] = category
            predicate_dict['supplier'][0]['value'] = nation
            predicate_dict['dwdate'][0]['value'] = [year_low, year_high]
        
        elif template_num == 14:
            linenumber_stats = self.stats['lineorder']['lo_linenumber']
            quantity_stats = self.stats['lineorder']['lo_quantity']

            linenumber_low = random.randint(linenumber_stats['min'], linenumber_stats['max']-2)
            linenumber_high = linenumber_low + 1
            quantity = random.randint(quantity_stats['min'], quantity_stats['max'])

            query = template.format(linenumber_low=linenumber_low, linenumber_high=linenumber_high, quantity=quantity)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['lineorder'][0]['value'] = (linenumber_low, linenumber_high)
            predicate_dict['lineorder'][1]['value'] = quantity

        elif template_num == 15:
            linenumber_stats = self.stats['lineorder']['lo_linenumber']
            quantity_stats = self.stats['lineorder']['lo_quantity']

            linenumber_low = random.randint(linenumber_stats['min'], linenumber_stats['max']-2)
            linenumber_high = linenumber_low + 1
            quantity = random.randint(quantity_stats['min'], quantity_stats['max'])

            query = template.format(linenumber_low=linenumber_low, linenumber_high=linenumber_high, quantity=quantity) 
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['lineorder'][0]['value'] = (linenumber_low, linenumber_high)
            predicate_dict['lineorder'][1]['value'] = quantity

        elif template_num == 16:
            extendedprice_stats = self.stats['lineorder']['lo_extendedprice']

            extendedprice = random.randint(extendedprice_stats['min'], extendedprice_stats['max'])

            query = template.format(extendedprice=extendedprice)
            predicate_dict = self.predicate_dicts[template_num]
            predicate_dict['lineorder'][0]['value'] = extendedprice

        # create Query object
        query = Query(template_num, query, self.payloads[template_num], self.predicates[template_num], self.order_bys[template_num], self.group_bys[template_num])
        query.predicate_dict = predicate_dict    

        return query
        

























