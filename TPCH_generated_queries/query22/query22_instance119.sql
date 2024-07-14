-- using 1720735369 as a seed to the RNG


select
	cntrycode,
	count(*) as numcust,
	sum(c_acctbal) as totacctbal
from
	(
		select
			substring(c_phone, 1, 2) as cntrycode,
			c_acctbal
		from
			customer
		where
			substring(c_phone, 1, 2) in
				('41', '23', '35', '34', '28', '42', '20')
			and c_acctbal > (
				select
					avg(c_acctbal)
				from
					customer
				where
					c_acctbal > 0.00
					and substring(c_phone, 1, 2) in
						('41', '23', '35', '34', '28', '42', '20')
			)
			and not exists (
				select
					*
				from
					orders
				where
					o_custkey = c_custkey
			)
	) as custsale
group by
	cntrycode
order by
	cntrycode;