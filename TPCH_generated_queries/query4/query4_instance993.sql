-- using 1720735353 as a seed to the RNG


select
	o_orderpriority,
	count(*) as order_count
from
	orders
where
	o_orderdate >= CAST('1994-09-01' AS date)
	and o_orderdate < DATEADD(mm, 3, CAST('1994-09-01' AS date))
	and exists (
		select
			*
		from
			lineitem
		where
			l_orderkey = o_orderkey
			and l_commitdate < l_receiptdate
	)
group by
	o_orderpriority
order by
	o_orderpriority;
