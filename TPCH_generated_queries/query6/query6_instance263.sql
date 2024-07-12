-- using 1720735354 as a seed to the RNG


select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= CAST('1994-01-01' AS date)
	and l_shipdate < DATEADD(yy, 1, CAST('1994-01-01' AS date))
	and l_discount between 0.09 - 0.01 and 0.09 + 0.01
	and l_quantity < 24;
