"""
    SSB query generator class
"""
import pickle
import random
import math


# returns SSB schema as a dictionary
def get_ssb_schema():
    ssb_schema = {
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
    return ssb_schema


# class for SSB queries
class Query:
    def __init__(self, template_id, query_string, payload, predicates, order_by, group_by):
        self.template_id = template_id
        self.query_string = query_string
        self.payload = payload
        self.predicates = predicates
        self.order_by = order_by
        self.group_by = group_by

    def __str__(self):
        return f"template id: {self.template_id}, query: {self.query_string}, payload: {self.payload}, predicates: {self.predicates}, order by: {self.order_by}, group by: {self.group_by}"    


# class for generating SSB queries
class QGEN:

    def __init__(self):
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
                AND lo_quantity < {quantity};
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
            """ 
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
                   "part": ["p_partkey", "p_mfgr", "p_category"], "supplier": ["s_suppkey", "s_nation"]}}

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
                        "supplier": ["s_city"]}}

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
                    13: {"dwdate": ["d_year"], "part": ["p_brand"], "supplier": ["s_city"]}}

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
                    13: {"part": ["p_brand"], "supplier": ["s_city"], "dwdate": ["d_year"]}}



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
            quantity = random.randint(quantity_stats['min'], quantity_stats['max'])

            query = template.format(year=year, discount_low=discount_low, discount_high=discount_high, quantity=quantity)

        elif template_num == 2:
            yearmonthnum_stats = self.stats['dwdate']['d_yearmonthnum']
            discount_stats = self.stats['lineorder']['lo_discount']
            quantity_stats = self.stats['lineorder']['lo_quantity']

            yearmonthnum = random.choice(list(yearmonthnum_stats['histogram'].keys()))
            discount_low = math.floor(random.uniform(float(discount_stats['min']), float(discount_stats['max'])-2))
            discount_high = discount_low + 2
            quantity_low = random.randint(quantity_stats['min'], quantity_stats['max'] // 2)
            quantity_high = random.randint(quantity_stats['max'] // 2, quantity_stats['max'])

            query = template.format(yearmonthnum=yearmonthnum, discount_low=discount_low, discount_high=discount_high, quantity_low=quantity_low, quantity_high=quantity_high)

        elif template_num == 3:
            weeknuminyear_stats = self.stats['dwdate']['d_weeknuminyear']
            year_stats = self.stats['dwdate']['d_year']
            discount_stats = self.stats['lineorder']['lo_discount']
            quantity_stats = self.stats['lineorder']['lo_quantity']

            weeknuminyear = random.choice(list(weeknuminyear_stats['histogram'].keys()))
            year = random.choice(list(year_stats['histogram'].keys()))
            discount_low = math.floor(random.uniform(float(discount_stats['min']), float(discount_stats['max'])-2))
            discount_high = discount_low + 2
            quantity_low = random.randint(quantity_stats['min'], quantity_stats['max'] // 2)
            quantity_high = random.randint(quantity_stats['max'] // 2, quantity_stats['max'])

            query = template.format(weeknuminyear=weeknuminyear, year=year, discount_low=discount_low, discount_high=discount_high, quantity_low=quantity_low, quantity_high=quantity_high)

        elif template_num == 4:
            category_stats = self.stats['part']['p_category']
            sregion_stats = self.stats['supplier']['s_region']

            category = random.choice(list(category_stats['histogram'].keys()))
            sregion = random.choice(list(sregion_stats['histogram'].keys()))

            query = template.format(category=category, sregion=sregion)

        elif template_num == 5:
            brand1_stats = self.stats['part']['p_brand']
            sregion_stats = self.stats['supplier']['s_region']

            brand1_values = list(brand1_stats['histogram'].keys())
            brand1_low = random.choice(brand1_values)
            brand1_high = random.choice([b for b in brand1_values if b >= brand1_low])
            sregion = random.choice(list(sregion_stats['histogram'].keys()))

            query = template.format(brand1_low=brand1_low, brand1_high=brand1_high, sregion=sregion)

        elif template_num == 6:
            brand1_stats = self.stats['part']['p_brand']
            sregion_stats = self.stats['supplier']['s_region']

            brand1 = random.choice(list(brand1_stats['histogram'].keys()))
            sregion = random.choice(list(sregion_stats['histogram'].keys()))

            query = template.format(brand1=brand1, sregion=sregion)

        elif template_num == 7:
            region_stats = self.stats['customer']['c_region']
            year_stats = self.stats['dwdate']['d_year']

            region = random.choice(list(region_stats['histogram'].keys()))
            year_low = random.choice(list(year_stats['histogram'].keys()))
            year_high = random.choice([y for y in year_stats['histogram'].keys() if y >= year_low])

            query = template.format(region=region, year_low=year_low, year_high=year_high)

        elif template_num == 8:
            region_stats = self.stats['customer']['c_nation']
            year_stats = self.stats['dwdate']['d_year']

            region = random.choice(list(region_stats['histogram'].keys()))
            year_low = random.choice(list(year_stats['histogram'].keys()))
            year_high = random.choice([y for y in year_stats['histogram'].keys() if y >= year_low])

            query = template.format(region=region, year_low=year_low, year_high=year_high)

        elif template_num == 9:
            city_stats = self.stats['customer']['c_city']
            year_stats = self.stats['dwdate']['d_year']

            city_1 = random.choice(list(city_stats['histogram'].keys()))
            city_2 = random.choice([c for c in city_stats['histogram'].keys() if c != city_1])
            year_low = random.choice(list(year_stats['histogram'].keys()))
            year_high = random.choice([y for y in year_stats['histogram'].keys() if y >= year_low])

            query = template.format(city_1=city_1, city_2=city_2, year_low=year_low, year_high=year_high)

        elif template_num == 10:
            city_stats = self.stats['customer']['c_city']
            yearmonth_stats = self.stats['dwdate']['d_yearmonth']

            city_1 = random.choice(list(city_stats['histogram'].keys()))
            city_2 = random.choice([c for c in city_stats['histogram'].keys() if c != city_1])
            yearmonth = random.choice(list(yearmonth_stats['histogram'].keys()))

            query = template.format(city_1=city_1, city_2=city_2, yearmonth=yearmonth)

        elif template_num == 11:
            region_stats = self.stats['customer']['c_region']
            mfgr_stats = self.stats['part']['p_mfgr']

            region = random.choice(list(region_stats['histogram'].keys()))
            mfgr_1 = random.choice(list(mfgr_stats['histogram'].keys()))
            mfgr_2 = random.choice([m for m in mfgr_stats['histogram'].keys() if m != mfgr_1])

            query = template.format(region=region, mfgr_1=mfgr_1, mfgr_2=mfgr_2)

        elif template_num == 12:
            region_stats = self.stats['customer']['c_region']
            year_stats = self.stats['dwdate']['d_year']
            mfgr_stats = self.stats['part']['p_mfgr']

            region = random.choice(list(region_stats['histogram'].keys()))
            year_1 = random.choice(list(year_stats['histogram'].keys()))
            year_2 = random.choice([y for y in year_stats['histogram'].keys() if y != year_1])
            mfgr_1 = random.choice(list(mfgr_stats['histogram'].keys()))
            mfgr_2 = random.choice([m for m in mfgr_stats['histogram'].keys() if m != mfgr_1])

            query = template.format(region=region, year_1=year_1, year_2=year_2, mfgr_1=mfgr_1, mfgr_2=mfgr_2)

        elif template_num == 13:
            region_stats = self.stats['customer']['c_region']
            nation_stats = self.stats['supplier']['s_nation']
            year_stats = self.stats['dwdate']['d_year']
            category_stats = self.stats['part']['p_category']

            region = random.choice(list(region_stats['histogram'].keys()))
            nation = random.choice(list(nation_stats['histogram'].keys()))
            year_1 = random.choice(list(year_stats['histogram'].keys()))
            year_2 = random.choice([y for y in year_stats['histogram'].keys() if y != year_1])
            category = random.choice(list(category_stats['histogram'].keys()))

            query = template.format(region=region, nation=nation, year_1=year_1, year_2=year_2, category=category)

        # create Query object
        query = Query(template_num, query, self.payloads[template_num], self.predicates[template_num], self.order_bys[template_num], self.group_bys[template_num])

        return query
        

























