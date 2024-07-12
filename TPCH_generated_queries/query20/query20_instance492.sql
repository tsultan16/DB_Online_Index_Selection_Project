-- using 1720735368 as a seed to the RNG


select
	s_name,
	s_address
from
	supplier,
	nation
where
	s_suppkey in (
		select
			ps_suppkey
		from
			partsupp
		where
			ps_partkey in (
				select
					p_partkey
				from
					part
				where
					p_name like 'spring%'
			)
			and ps_availqty > (
				select
					0.5 * sum(l_quantity)
				from
					lineitem
				where
					l_partkey = ps_partkey
					and l_suppkey = ps_suppkey
					and l_shipdate >= CAST('1997-01-01' AS date)
					and l_shipdate < DATEADD(yy, 1, CAST('1997-01-01' AS date))
			)
	)
	and s_nationkey = n_nationkey
	and n_name = 'GERMANY'
order by
	s_name;
