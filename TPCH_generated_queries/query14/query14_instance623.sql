-- using 1720735362 as a seed to the RNG


select
	100.00 * sum(case
		when p_type like 'PROMO%'
			then l_extendedprice * (1 - l_discount)
		else 0
	end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
	lineitem,
	part
where
	l_partkey = p_partkey
	and l_shipdate >= CAST('1997-08-01' AS date)
	and l_shipdate < DATEADD(mm, 1, CAST('1997-08-01' AS date));