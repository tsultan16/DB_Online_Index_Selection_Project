"""
    WFIT Implementation
"""


import sys
import os

# Add Postgres directory to sys.path
module_path = os.path.abspath('/home/tanzid/Code/DBMS/PostgreSQL')
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath('/home/tanzid/Code/DBMS/WFIT')
if module_path not in sys.path:
    sys.path.append(module_path)

from pg_utils import *
from ssb_qgen_class import *
from simple_cost_model import *

from collections import defaultdict, deque
import time
import random
from more_itertools import powerset
from itertools import chain, permutations
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import math

#from functools import lru_cache


DBNAME = 'tpch10'


"""
    Index Benefit Graph
"""

class Node:
    def __init__(self, id, indexes):
        self.id = id
        self.indexes = indexes
        self.children = []
        self.parents = []
        self.built = False
        self.cost = None
        self.used = []


# class for creating and storing the IBG
class IBG:
    # Class-level cache
    _pk_indexes = None

    def __init__(self, query_object, C, existing_indexes=[], execution_cost_scaling=1e-6, ibg_max_nodes=100, doi_max_nodes=50, max_doi_iters_per_node=100, normalize_doi=True, simple_cost_model=None):
        self.q = query_object
        self.C = C
        self.existing_indexes = existing_indexes # indexes currently materialized in the database
        self.execution_cost_scaling = execution_cost_scaling
        self.ibg_max_nodes = ibg_max_nodes
        self.doi_max_nodes = doi_max_nodes
        self.max_doi_iters_per_node = max_doi_iters_per_node
        self.normalize_doi = normalize_doi
        print(f"Number of candidate indexes: {len(self.C)}")
        #print(f"Candidate indexes: {self.C}")
        
        if IBG._pk_indexes is None:
            self.pk_indexes = tpch_pk_index_objects()

        if simple_cost_model is not None:
            self.simple_cost = True
            self.simple_cost_model = simple_cost_model
        else:
            self.simple_cost = False    

        # create a connection session to the database
        self.conn = create_connection(dbname=DBNAME)
        # get hypothetical sizes of all the candidate indexes
        print("Getting hypothetical sizes of candidate indexes...")
        self.get_hypo_sizes()
        # hide existing indexes
        bulk_hide_indexes(self.conn, self.existing_indexes)
        
        # map index_id to integer
        self.idx2id = {index.index_id:i for i, index in enumerate(self.C)}
        self.idx2index = {index.index_id:index for index in self.C}
        #print(f"Index id to integer mapping: {self.idx2id}")
        
        # create a hash table for keeping track of all created nodes
        self.nodes = {}
        # create a root node
        self.root = Node(self.get_configuration_id(self.C), self.C)
        self.nodes[self.root.id] = self.root
        print(f"Created root node with id: {self.root.id}")
        
        self.total_whatif_calls = 0
        self.total_whatif_time = 0
        self.node_count = 0

        # start the IBG construction
        start_time = time.time()
        print("Constructing IBG...")
        self.construct_ibg(self.root, max_nodes=ibg_max_nodes)  # truncate IBG at ibg_max_nodes
        end_time = time.time()
        ibg_construction_time = end_time - start_time
        print(f"Number of nodes in IBG: {len(self.nodes)}, Total number of what-if calls: {self.total_whatif_calls}, Time spent on what-if calls: {self.total_whatif_time}, IBG construction time: {ibg_construction_time}")

        """
        # compute all pair degree of interaction
        print(f"Computing all pair degree of interaction...")
        start_time = time.time()
        #self.doi = self.compute_all_pair_doi()
        
        #self.doi = self.compute_all_pair_doi_parallel(num_workers=16, max_nodes=doi_max_nodes, max_iters_per_node=max_doi_iters_per_node)
        
        #self.doi = self.compute_all_pair_doi_simple()
        #self.doi = self.compute_all_pair_doi_naive(num_samples=32)
        
        #self.doi = self.compute_all_pair_doi_naive_parallel(num_samples=8, num_workers=4, max_iters=max_doi_iters_per_node)
        self.doi = self.compute_all_pair_doi_naive_parallel(max_iters=doi_max_nodes, num_workers=16)
        
        #print(f"All pair doi:")
        #for key, value in self.doi.items():
        #    print(f"{key}: {value}")
        end_time = time.time()
        print(f"\nTime spent on computing all pair degree of interaction: {end_time - start_time}\n")
        """

        # unhide existing indexes
        bulk_unhide_indexes(self.conn, self.existing_indexes)
        close_connection(self.conn)
        self.conn = None


    def compute_doi(self):
      # create a connection session to the database
        self.conn = create_connection(dbname=DBNAME)
        # hide existing indexes
        bulk_hide_indexes(self.conn, self.existing_indexes)
        
        # compute all pair degree of interaction
        print(f"Computing all pair degree of interaction...")
        start_time = time.time()
    
        #self.doi = self.compute_all_pair_doi_parallel(num_workers=16, max_nodes=self.doi_max_nodes, max_iters_per_node=self.max_doi_iters_per_node)
        self.doi = self.compute_all_pair_doi_naive_parallel(max_iters=self.doi_max_nodes, num_workers=16)
        
        #print(f"All pair doi:")
        #for key, value in self.doi.items():
        #    print(f"{key}: {value}")
        end_time = time.time()
        print(f"\nTime spent on computing all pair degree of interaction: {end_time - start_time}\n")

        # unhide existing indexes
        bulk_unhide_indexes(self.conn, self.existing_indexes)
        close_connection(self.conn)
        self.conn = None


    # assign unique string id to a configuration
    def get_configuration_id(self, indexes):
        # get sorted list of integer ids
        ids = sorted([self.idx2id[idx.index_id] for idx in indexes])
        return "_".join([str(i) for i in ids])
    

    # get hypothetical sizes of all the candidate indexes
    def get_hypo_sizes(self):
        hypo_indexes = bulk_create_hypothetical_indexes(self.conn, self.C, return_size=True)
        for i in range(len(hypo_indexes)):
            self.C[i].size = hypo_indexes[i][1]


    #@lru_cache(maxsize=None)
    def _get_cost_used_1(self, indexes):
       
        start_time = time.time()
        if self.conn is None:
            conn = create_connection(dbname=DBNAME)
        else:
            conn = self.conn    
        # create hypothetical indexes
        hypo_indexes = bulk_create_hypothetical_indexes(conn, indexes)
        # map oid to index object
        oid2index = {}
        for i in range(len(hypo_indexes)):
            oid2index[hypo_indexes[i]] = indexes[i]
        # get cost and used indexes
        cost, indexes_used = get_query_cost_estimate_hypo_indexes(conn, self.q.query_string, show_plan=False, get_secondary_indexes_used=True)
        # map used index oids to index objects
        used = []
        for oid, scan_type, scan_cost in indexes_used:
                used.append(oid2index[oid])
            
        # drop hypothetical indexes
        bulk_drop_hypothetical_indexes(conn)
        if self.conn is None:
            close_connection(conn)
        end_time = time.time()

        # Store the result in the class-level cache
        #self._class_cache[indexes_tuple] = (cost, used)
        self.total_whatif_calls += 1
        self.total_whatif_time += end_time - start_time

        #print(f"Configuration: {[index.index_id for index in indexes]}, Cost: {cost}, Used indexes: {[index.index_id for index in used]}")
        # scale the execution cost
        cost *= self.execution_cost_scaling    
        return cost, used


    def _get_cost_used_2(self, indexes):
        indexes = {index.index_id:index for index in indexes + self.pk_indexes}
        cost, used = self.simple_cost_model.predict(indexes, verbose=False)
        #print(f"Number of configuration indexes:{len(indexes)}, Cost: {cost}, Used indexes: {used}")
        cost *= self.execution_cost_scaling
        used = [self.idx2index[index_id] for index_id in used]
        return cost, used


    # Ensure the indexes parameter is hashable
    def _cached_get_cost_used(self, indexes):
        if not self.simple_cost:
            return self._get_cost_used_1(tuple(indexes))
        else:
            return self._get_cost_used_2(indexes)
    

    # IBG construction
    def construct_ibg(self, root, max_nodes=None):
        # Obtain query optimizer's cost and used indexes
        cost, used = self._cached_get_cost_used(root.indexes)
        #cost, used = self._get_cost_used(root.indexes)
        root.cost = cost
        root.used = used
        root.built = True
        self.node_count += 1

        num_levels = 0
        queue = deque([root])
        while queue:
            if max_nodes is not None and self.node_count >= max_nodes:
                break  # end if the maximum number of nodes is reached
            
            # Get the current level size
            level_size = len(queue)
            num_levels += 1

            # Process all nodes at the current level
            for _ in range(level_size):
                Y = queue.popleft()
                               
                # Create children
                for a in Y.used:
                    # Create a new configuration with index a removed from Y
                    X_indexes = [index for index in Y.indexes if index != a]
                    X_id = self.get_configuration_id(X_indexes)
                    
                    # If X is not in the hash table, create a new node and add it to the queue
                    if X_id not in self.nodes:
                        self.node_count += 1
                        #print(f"Creating node # {self.node_count}", end="\r")
         
                        X = Node(X_id, X_indexes)
                        # Obtain query optimizer's cost and used indexes
                        cost, used = self._cached_get_cost_used(X.indexes)
                        #cost, used = self._get_cost_used(X.indexes)
                        X.cost = cost
                        X.used = used
                        X.built = True
                        X.parents.append(Y)
                        self.nodes[X_id] = X
                        Y.children.append(X)
                        queue.append(X)

                    else:
                        X = self.nodes[X_id]
                        Y.children.append(X)
                        X.parents.append(Y)

        print(f"Number of levels in IBG: {num_levels}")      


    # use IBG to obtain estimated cost and used indexes for arbitrary subset of C
    def get_cost_used(self, X):
        # get id of the configuration
        id = self.get_configuration_id(X)
        # check if the configuration is in the IBG
        if id in self.nodes:
            cost, used = self.nodes[id].cost, self.nodes[id].used
        
        # if not in the IBG, traverse the IBG to find a covering node
        else:
            Y = self.find_covering_node(X)              
            cost, used = Y.cost, Y.used
        
        return cost, used    


    # traverses the IBG to find a node that removes indexes not in X (i.e. a covering node for X)
    def find_covering_node(self, X):
        X_indexes = set([index.index_id for index in X])
        Y = self.root
        Y_indexes = set([index.index_id for index in Y.indexes])
        # traverse IBG to find covering node
        while len(Y.children) > 0:               
            # traverse down to the child node that removes an index not in X
            child_found = False
            for child in Y.children:
                child_indexes = set([index.index_id for index in child.indexes])
                child_indexes_removed = Y_indexes - child_indexes
                child_indexes_removed_not_in_X = child_indexes_removed - X_indexes
        
                # check if child removes an index not in X
                if len(child_indexes_removed_not_in_X) > 0:
                    Y = child
                    Y_indexes = child_indexes
                    child_found = True
                    break

            # if no children remove indexes not in X    
            if not child_found:
                break    
    
        return Y        

    # compute benefit of an index for a given configuration 
    # input X is a list of index objects and 'a' is a single index object
    # X must not contain 'a'
    def compute_benefit(self, a, X):
        if a in X:
            # zero benefit if 'a' is already in X
            #raise ValueError("Index 'a' is already in X")
            return 0          

        #if self.simple_cost:
        #    cost_X = self._get_cost_used_2(X)[0]
        #    cost_X_a = self._get_cost_used_2(X + [a])[0]

        #else:    
        
        # get cost  for X
        cost_X = self.get_cost_used(X)[0]
        # create a new configuration with index a added to X
        X_a = X + [a]
        # get cost for X + {a}
        cost_X_a = self.get_cost_used(X_a)[0]

        # compute benefit
        benefit = cost_X - cost_X_a

        return benefit 


    # compute maximum benefit of adding an index to any possibe configuration
    def compute_max_benefit(self, a):
        max_benefit = float('-inf')

        # separately compute empty set configuration benefit
        #benefit = self.compute_benefit(a, [])
        #if benefit > max_benefit:
        #    max_benefit = benefit

        for id, node in self.nodes.items():
            #print(f"Computing benefit for node: {[index.index_id for index in node.indexes]}")
            benefit = self.compute_benefit(a, node.indexes)
            if benefit > max_benefit:
                max_benefit = benefit

        return max_benefit
    
    # compute the degree of interaction between two indexes a,b in configuration X 
    def compute_doi_configuration(self, a, b, X=[]):
        # X must not contain a or b
        if a in X or b in X:
            raise ValueError("a or b is already in X")

        doi = abs(self.compute_benefit(a, X) - self.compute_benefit(a, X + [b]))
        if self.normalize_doi:
            doi /= self.get_cost_used(X + [a,b])[0]   
        return doi
   
    
    # Cache the results of find_covering_node and get_cost_used to avoid redundant calculations
    #@lru_cache(maxsize=None)
    def cached_find_covering_node(self, indexes):
        return self.find_covering_node(tuple(indexes))

    #@lru_cache(maxsize=None)
    def cached_get_cost_used(self, indexes):
        return self.get_cost_used(tuple(indexes))


    # computes the degree of interaction between all pairs of indexes (a,b) in candidate set C
    # Note: doi is symmetric, i.e. doi(a,b) = doi(b,a)

    # simple version of compute_all_pair_doi, without parallelization
    def compute_all_pair_doi_simple(self):
        # hash table for storing doi values
        doi = {}
        # intialize doi values to zero
        for i in range(len(self.C)):
            for j in range(i+1, len(self.C)):
                d = self.compute_doi_configuration(self.C[i], self.C[j])
                doi[tuple(sorted((self.C[i].index_id, self.C[j].index_id)))] = d

        return doi


    # Naive version of compute_all_pair_doi, with random sampling of configurations
    def compute_all_pair_doi_naive(self, num_samples=100):
        doi = {}
        
        for i in range(len(self.C)):
            for j in range(i + 1, len(self.C)):
                doi[tuple(sorted((self.C[i].index_id, self.C[j].index_id)))] = 0
        
        # sample random configurations: X subset C (must include empty set configuration)
        for i in tqdm(range(num_samples), desc="Sampling configurations"):
            if i == 0:
                X = []
            else:
                X = random.sample(self.C, random.randint(1, len(self.C)))

            # compute doi for all pairs (a, b) in U\X 
            for i in range(len(self.C)):
                for j in range(i+1, len(self.C)):
                    a = self.C[i]
                    b = self.C[j]
                    if a not in X and b not in X:
                        d = self.compute_doi_configuration(a, b, X)
                        key = tuple(sorted((a.index_id, b.index_id)))
                        doi[key] = max(doi[key], d)
        
        return doi    


    # Naive version of compute_all_pair_doi, with random sampling of configurations, parallelized
    def compute_all_pair_doi_naive_parallel(self, max_iters, num_workers=8):
        doi = {}
        
        # Initialize DOI dictionary with zero values
        for i in range(len(self.C)):
            for j in range(i + 1, len(self.C)):
                doi[tuple(sorted((self.C[i].index_id, self.C[j].index_id)))] = 0

        # get all the IBG node configurations
        IBG_configs = [node.indexes for node in self.nodes.values()]

        # sample max_nodes number of nodes from the chunk
        unique_X_configs = random.sample(IBG_configs, min(max_iters, len(IBG_configs))) 
        # add empty set configuration
        unique_X_configs.append([])

        # Distribute configurations evenly among num_workers
        chunks = [unique_X_configs[i::num_workers] for i in range(num_workers)]

        def compute_doi_for_chunk(chunk):
            local_doi = {}
            for X in chunk:
                C_sample = self.C  # Assuming you want to work over the entire set C for each X
                for i in range(len(C_sample)):
                    for j in range(i + 1, len(C_sample)):
                        a = C_sample[i]
                        b = C_sample[j]
                        if a not in X and b not in X:
                            d = self.compute_doi_configuration(a, b, X)
                            key = tuple(sorted((a.index_id, b.index_id)))
                            if key not in local_doi:
                                local_doi[key] = d
                            else:
                                local_doi[key] = max(local_doi[key], d)
            return local_doi

        # Use ThreadPoolExecutor for parallel execution with specified number of workers
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(compute_doi_for_chunk, chunk): chunk for chunk in chunks}
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Processing chunks"):
                local_doi = future.result()
                for key, value in local_doi.items():
                    doi[key] = max(doi[key], value)

        return doi


    # original version of compute_all_pair_doi, with optional max_nodes parameter for random sampling of nodes for efficient approximation
    def compute_all_pair_doi(self, max_nodes=None):
        # hash table for storing doi values
        doi = {}
        # intialize doi values to zero
        for i in range(len(self.C)):
            for j in range(i+1, len(self.C)):
                doi[tuple(sorted((self.C[i].index_id, self.C[j].index_id)))] = 0

        S_idxs = set([index.index_id for index in self.C])

        # sample max_nodes number of nodes from the chunk
        if max_nodes is not None:
            nodes_sample = random.sample(self.nodes.values(), min(max_nodes, len(self.nodes)))
        else:
            nodes_sample = self.nodes.values()

        # iterate over each IBG node
        for Y in tqdm(nodes_sample, desc="Processing nodes"):
            
            # remove Y.used from S
            Y_idxs = set([index.index_id for index in Y.indexes])
            used_Y = Y.used
            Y_used_idxs = set([index.index_id for index in used_Y])
            S_Y = list(S_idxs - Y_used_idxs)
            # iterate over all pairs of indexes in S_Y
            for i in range(len(S_Y)):
                for j in range(i+1, len(S_Y)):
                    a_idx = S_Y[i]
                    b_idx = S_Y[j]
                     
                    # find Ya covering node in IBG
                    Ya = (Y_idxs - {a_idx, b_idx}) | {a_idx}
                    Ya = [self.idx2index[idx] for idx in Ya]
                    Ya = self.cached_find_covering_node(tuple(Ya))
                    # find Yab covering node in IBG
                    Yab = (Y_idxs - {a_idx, b_idx}) | {a_idx, b_idx}
                    Yab = [self.idx2index[idx] for idx in Yab]
                    Yab = self.cached_find_covering_node(tuple(Yab))

                    #used_Y = self.cached_get_cost_used(tuple(Y.indexes))[1]
                    #used_Ya = self.cached_get_cost_used(tuple(Ya))[1]
                    #used_Yab = self.cached_get_cost_used(tuple(Yab))[1]
                    used_Ya = Ya.used
                    used_Yab = Yab.used

                    Uab = set([index.index_id for index in used_Y]) | set([index.index_id for index in used_Ya]) | set([index.index_id for index in used_Yab]) 
                    # find Yb_minus covering node in IBG 
                    Yb_minus = list((Uab - {a_idx, b_idx}) | {b_idx})
                    Yb_minus = [self.idx2index[idx] for idx in Yb_minus]
                    Yb_minus = self.cached_find_covering_node(tuple(Yb_minus))
                    # find Yb_plus covering node in IBG
                    Yb_plus = list((Y_idxs - {a_idx, b_idx}) | {b_idx})
                    Yb_plus = [self.idx2index[idx] for idx in Yb_plus]
                    Yb_plus = self.cached_find_covering_node(tuple(Yb_plus))

                    # generate quadruples
                    quadruples = [(Y.indexes, Ya.indexes, Yb_minus.indexes, Yab.indexes), (Y.indexes, Ya.indexes, Yb_plus.indexes, Yab.indexes)]

                    # compute doi using the quadruples
                    for Y_indexes, Ya_indexes, Yb_indexes, Yab_indexes in quadruples:
                        cost_Y = self.cached_get_cost_used(tuple(Y_indexes))[0]
                        cost_Ya = self.cached_get_cost_used(tuple(Ya_indexes))[0]
                        cost_Yb = self.cached_get_cost_used(tuple(Yb_indexes))[0]
                        cost_Yab = self.cached_get_cost_used(tuple(Yab_indexes))[0]
                        # can ignore the normalization terms in denominator to get an absolute measure of doi
                        d = abs(cost_Y - cost_Ya - cost_Yb + cost_Yab) 
                        if self.normalize_doi: 
                            d /= cost_Yab
                        # save doi value for the pair
                        key = tuple(sorted((a_idx, b_idx)))
                        doi[key] = max(doi[key], d)
                            
        return doi


    # parallelized version of compute_all_pair_doi
    def compute_all_pair_doi_parallel(self, num_workers=16, max_nodes=None, max_iters_per_node=None):
        doi = {}
        
        for i in range(len(self.C)):
            for j in range(i + 1, len(self.C)):
                doi[tuple(sorted((self.C[i].index_id, self.C[j].index_id)))] = 0
        
        S_idxs = set([index.index_id for index in self.C])
        
        if max_nodes is not None:
            nodes_list = random.sample(list(self.nodes.values()), min(max_nodes, len(self.nodes)))
        else:    
            nodes_list = list(self.nodes.values())
        
        chunk_size = max(1, len(nodes_list) // num_workers)

        chunks = [nodes_list[i:i + chunk_size] for i in range(0, len(nodes_list), chunk_size)]
        
        args = [(chunk, self.C, self.idx2index, S_idxs, self.cached_find_covering_node, self.cached_get_cost_used, self.normalize_doi, max_iters_per_node) for chunk in chunks]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_node_chunk, args), total=len(chunks), desc="Processing nodes in parallel"))
        
        for result in results:
            for key, value in result.items():
                doi[key] = max(doi.get(key, 0), value)
        
        return doi
    
    
    # get precomputed degree of interaction between a pair of indexes
    def get_doi_pair(self, a, b):
            return self.doi[tuple(sorted((a.index_id, b.index_id)))]


    # function for printing the IBG, using BFS level order traversal
    def print_ibg(self):
        q = [self.root]
        # traverse level by level, print all node ids in a level in a single line before moving to the next level
        while len(q) > 0:
            next_q = []
            for node in q:
                print(f"{node.id} -> ", end="")
                for child in node.children:
                    next_q.append(child)
            print()
            q = next_q  


def process_node_chunk(args):
    nodes_chunk, C, idx2index, S_idxs, cached_find_covering_node, cached_get_cost_used, normalize_doi, max_iters_per_node = args
    doi_chunk = {}
    
    for Y in nodes_chunk:
        Y_idxs = set([index.index_id for index in Y.indexes])
        used_Y = Y.used
        Y_used_idxs = set([index.index_id for index in used_Y])
        S_Y = list(S_idxs - Y_used_idxs)
        
        iter_count = 0
        for i in range(len(S_Y)):
            for j in range(i + 1, len(S_Y)):
                iter_count += 1
                a_idx = S_Y[i]
                b_idx = S_Y[j]
                
                Ya = (Y_idxs - {a_idx, b_idx}) | {a_idx}
                Ya = [idx2index[idx] for idx in Ya]
                Ya = cached_find_covering_node(tuple(Ya))
                
                Yab = (Y_idxs - {a_idx, b_idx}) | {a_idx, b_idx}
                Yab = [idx2index[idx] for idx in Yab]
                Yab = cached_find_covering_node(tuple(Yab))
                
                used_Ya = Ya.used
                used_Yab = Yab.used
                
                Uab = set([index.index_id for index in used_Y]) | set([index.index_id for index in used_Ya]) | set([index.index_id for index in used_Yab])
                
                Yb_minus = list((Uab - {a_idx, b_idx}) | {b_idx})
                Yb_minus = [idx2index[idx] for idx in Yb_minus]
                Yb_minus = cached_find_covering_node(tuple(Yb_minus))
                
                Yb_plus = list((Y_idxs - {a_idx, b_idx}) | {b_idx})
                Yb_plus = [idx2index[idx] for idx in Yb_plus]
                Yb_plus = cached_find_covering_node(tuple(Yb_plus))
                
                quadruples = [(Y.indexes, Ya.indexes, Yb_minus.indexes, Yab.indexes), (Y.indexes, Ya.indexes, Yb_plus.indexes, Yab.indexes)]
                
                for Y_indexes, Ya_indexes, Yb_indexes, Yab_indexes in quadruples:
                    cost_Y = cached_get_cost_used(tuple(Y_indexes))[0]
                    cost_Ya = cached_get_cost_used(tuple(Ya_indexes))[0]
                    cost_Yb = cached_get_cost_used(tuple(Yb_indexes))[0]
                    cost_Yab = cached_get_cost_used(tuple(Yab_indexes))[0]
                    
                    d = abs(cost_Y - cost_Ya - cost_Yb + cost_Yab)
                    if normalize_doi: 
                            d /= cost_Yab
                    key = tuple(sorted((a_idx, b_idx)))
                    doi_chunk[key] = max(doi_chunk.get(key, 0), d)
    
                if max_iters_per_node is not None and iter_count >= max_iters_per_node:
                    break
            if  max_iters_per_node is not None and iter_count >= max_iters_per_node:
                break

    return doi_chunk




"""
    WFIT Implementation
"""

class WFIT:
    database_size_cache = None
    _stats_cache = None
    _estimated_rows_cache = None

    def __init__(self, max_key_columns=None, include_cols=False, max_include_columns=3, simple_cost=False, enable_stable_partition_locking=True, max_indexes_per_table=3, max_U=100, ibg_max_nodes=100, doi_max_nodes=50, max_doi_iters_per_node=100, normalize_doi=True, idxCnt=25, stateCnt=500, histSize=1000, rand_cnt=100, execution_cost_scaling=1e-6, creation_cost_fudge_factor=1, join_column_discount=0.7):
        # bulk drop all materialized secondary indexes
        print(f"*** Dropping all materialized secondary indexes...")
        conn = create_connection(dbname=DBNAME)
        drop_all_indexes(conn)
        close_connection(conn)

        # get database size
        if WFIT.database_size_cache is None:
            conn = create_connection(dbname=DBNAME)
            WFIT.database_size_cache = get_database_size(conn)
            close_connection(conn)
        else:
            print("Found database size in cache")    
        self.database_size = WFIT.database_size_cache    
        print(f"Database size: {self.database_size} MB")

        if simple_cost:
            print(f"*** Using simple cost model...")

        # initial set of materialzed indexes
        #self.pk_indexes = ssb_pk_index_objects() 
        self.S_0 = [] #self.pk_indexes # only primary key indexes are materialized initially
        # maximum number of key columns in an index
        self.max_key_columns = max_key_columns
        # allow include columns in indexes
        self.include_cols = include_cols
        # maximum number of include columns in an index
        self.max_include_columns = max_include_columns
        # use simple cost model
        self.simple_cost = simple_cost
        # enable stable partition locking
        self.enable_stable_partition_locking = enable_stable_partition_locking
        # maximum number of indexes to monitor
        self.max_U = max_U

        # hard-coded 8 for tpch
        if max_U < max_indexes_per_table * 8:
            raise ValueError("max_U must be at least 8 times the maximum number of indexes per table") 

        # maximum number of nodes in IBG
        self.ibg_max_nodes = ibg_max_nodes
        # maximum number of nodes in DOI computation
        self.doi_max_nodes = doi_max_nodes
        # maximum number of iterations per node in DOI computation
        self.max_doi_iters_per_node = max_doi_iters_per_node
        # normalize degree of interaction
        self.normalize_doi = normalize_doi
        # parameter for maximum number of candidate indexes tracked 
        self.idxCnt = idxCnt
        # parameter for maximum number of MTS states/configurations
        self.stateCnt = stateCnt
        # parameter for maximum number of historical index statistics kept
        self.histSize = histSize
        # parameter for number of randomized clustering iterations
        self.rand_cnt = rand_cnt
        # fudge factor for index creation cost
        self.creation_cost_fudge_factor = creation_cost_fudge_factor
        # discount factor for join columns
        self.join_column_discount = join_column_discount
        # fudge factor for execution cost
        self.execution_cost_scaling = execution_cost_scaling
        # growing list of candidate indexes (initially contains S_0)
        self.U = {index.index_id:index for index in self.S_0}
        # index benefit and interaction statistics
        self.idxStats = defaultdict(list)
        self.intStats = defaultdict(list)
        # list of currently monitored indexes
        self.C = {index.index_id:index for index in self.S_0} 
        # list of currently materialized indexes
        self.M = {index.index_id:index for index in self.S_0}  
        # initialize stable partitions (each partition is a singleton set of indexes from S_0)
        self.stable_partitions = [[index] for index in self.S_0] if self.S_0 else [[]]
        # initialize current recommendations for each stable partition
        self.current_recommendations = {i:indexes for i, indexes in enumerate(self.stable_partitions)}
        # keep track of candidate index sizes
        self.index_size = {}
        self.n_pos = 0
        self.n_round = 0
        # index usage stats
        self.index_usage = defaultdict(int)

        """
        Stable partitions simple locking mechanism:

        If we stop seeing new query templates, then it is likely that the stable partitions are not changing. So, we can just lock the stable partitions
        so they stay fixed and avoid recomputing new candidate indexes, stable partitions, etc. Then the WFA updates over the same configuration space.

        Locking mechanism currently only compatible with simple cost model. Can't use with HypoPG what-if, would require updating IBG.
        """
        self.stable_partitions_locked = False
        self.query_templates_seen = set()
        self.ibg_prev = None

        print(f"##################################################################")
        # initialize work function instance for each stable partition
        self.W = self.initilize_WFA(self.stable_partitions)

        print(f"\nInitial set of materialized indexes: {[index.index_id for index in self.S_0]}")
        print(f"Stable partitions: {[[index.index_id for index in P] for P in self.stable_partitions]}")
        print(f"\nMaximum number of candidate indexes tracked: {idxCnt}")
        print(f"Maximum number of MTS states/configurations: {stateCnt}")
        print(f"Maximum number of historical index statistics kept: {histSize}")
        print(f"Number of randomized clustering iterations: {rand_cnt}")
        print(f"##################################################################\n")

        # set random seed
        random.seed(1234)
        # track time 
        self.batch_recommendation_time = []
        self.batch_materialization_time = []
        self.batch_execution_time = []
        self.execution_time = []
        self.configuration_stats = {"current_configuration": [], "indexes_added": [], "indexes_removed": []}
        self.total_recommendation_time = 0
        self.total_materialization_time = 0
        self.total_execution_time_actual = 0
        self.total_time_actual = 0
        self.total_cost_wfit = 0
        self.total_cost_simple = 0
        self.total_no_index_cost = 0

        # max number of indexes per table allowed in top-k indexes
        self.MAX_INDEXES_PER_TABLE = max_indexes_per_table


    # initialize a WFA instance for each stable partition
    def initilize_WFA(self, stable_partitions):
        print(f"Initializing WFA instances for {len(stable_partitions)} stable partitions...")
        W = {}
        for i, P in enumerate(stable_partitions):
            print(f"\tPartition #{i+1}: {[index.index_id for index in P]}")
            S_0 = self.current_recommendations[i]
            # initialize all MTS states, i.e. power set of indexes in the partition
            states = [tuple(sorted(state, key=lambda x: x.index_id)) for state in powerset(P)]
            # initialize work function instance for the partition
            W[i] = {}
            for X in states:
                sorted_X = tuple(sorted(X, key=lambda x: x.index_id))
                W[i][sorted_X] = self.compute_transition_cost(S_0, X) 
                print(f"\t\t w[{tuple([index.index_id for index in sorted_X])}]: {W[i][sorted_X]}")

        return W


    # update WFIT over next batch of queries (i.e. mini workload)
    def process_WFIT_batch(self, query_objects, restart_server=False, clear_cache=False, remove_stale_U=False, remove_stale_freq=1, materialize=True, execute=True, verbose=True):
        self.n_round += 1
        previous_config = list(self.M.values())
        # process each query in the workload 
        if verbose: print(f"Processing batch of queries...\n")
        start_time = time.time()
        for i, query_object in enumerate(query_objects):
            if verbose: 
                print(f"\nProcessing query ({i+1}/{len(query_objects)}) --> template_id: {query_object.template_id} ...")
                print('----------------------------------------------------')

            start_time_2 = time.time()
            self.n_pos += 1        
            # generate new partitions 
            if verbose: print(f"Generating new partitions for query #{self.n_pos}")
            start_time_1 = time.time()
            new_partitions, need_to_repartition, ibg = self.choose_candidates(self.n_pos, query_object, remove_stale_U, remove_stale_freq, verbose=verbose)
            end_time_1 = time.time()
            partitioning_time = end_time_1 - start_time_1
            # repartition if necessary
            start_time_1 = time.time()
            if need_to_repartition:
                if verbose: print(f"\nRepartitioning...\n")
                self.repartition(new_partitions, verbose=False)

            end_time_1 = time.time()
            repartitioning_time = end_time_1 - start_time_1
            # analyze the query and get recommendation
            if verbose: print(f"\n\nAnalyzing...\n")
            start_time_1 = time.time()
            all_indexes_added, all_indexes_removed = self.analyze_query(query_object, ibg, verbose=verbose)
            end_time_1 = time.time()
            analysis_time = end_time_1 - start_time_1
            end_time_2 = time.time()
            if verbose: 
                print(f"\nPartitioning time: {partitioning_time} s")
                print(f"Repartitioning time: {repartitioning_time} s")
                print(f"WFA update time: {analysis_time} s")
                print(f"Total recommendation time for query: {end_time_2 - start_time_2} s\n")

            if verbose:
                print(f"Recommendation Indexes added:   {[index.index_id for index in all_indexes_added]}")
                print(f"\nRecommendation Indexes removed: {[index.index_id for index in all_indexes_removed]}")

        end_time = time.time()
        recommendation_time = end_time - start_time
        if verbose: print(f"\nRecommendation time for batch of queries: {recommendation_time} s")

        # find out which indexes are added and removed at the end of the batch
        all_indexes_added = []
        all_indexes_removed = []
        for index in self.M.values():
            if index not in previous_config:
                all_indexes_added.append(index)
        for index in previous_config:
            if index not in self.M.values():
                all_indexes_removed.append(index)


        # materialize new configuration
        if materialize:
            config_materialization_time = self.materialize_configuration(all_indexes_added, all_indexes_removed, verbose)
        else:
            config_materialization_time = 0
        
        if verbose: 
            print(f"\nConfiguration materialization time: {config_materialization_time} s\n")
            print(f"{len(self.M)} currently materialized indexes: {[index.index_id for index in self.M.values()]}\n")
            print(f"\n{len(all_indexes_added)} indexes added this round: {[index.index_id for index in all_indexes_added]}")
            print(f"\n{len(all_indexes_removed)} indexes removed this round: {[index.index_id for index in all_indexes_removed]}")
            print(f"\nTotal Configuration Size: {sum([index.size for index in self.M.values()])} MB\n")

        self.configuration_stats["current_configuration"].append([index.index_id for index in self.M.values()])
        self.configuration_stats["indexes_added"].append([index.index_id for index in all_indexes_added])
        self.configuration_stats["indexes_removed"].append([index.index_id for index in all_indexes_removed])

        # restart the server before batch query execution
        #restart_postgresql()    

        # execute the batch of queries with the new configuration
        batch_execution_time = 0
        if execute:
            for i, query_object in enumerate(query_objects):
                if restart_server:
                    # restart the server before each query execution
                    restart_postgresql(clear_cache=clear_cache)
                if verbose: print(f"Executing query ({i+1}/{len(query_objects)})  --> template_id: {query_object.template_id} ...")
                conn = create_connection(dbname=DBNAME)
                execution_time, rows, table_access_info, index_access_info, bitmap_heapscan_info = execute_query(conn, query_object.query_string, with_explain=True, return_access_info=True)
                close_connection(conn)
                execution_time /= 1000
                batch_execution_time += execution_time
                self.execution_time.append(execution_time)
                if verbose:
                    #print(f"Indexes accessed --> {index_access_info}")
                    #print(f"Bitmap Heap Scans --> {bitmap_heapscan_info}") 
                    print(f"\tExecution_time: {execution_time} s, Tables accessed: {list(set(table_access_info.keys()).union(bitmap_heapscan_info.keys()))}, Indexes accessed: {list(index_access_info.keys())}\n")
                # update index usage stats
                for index_id in index_access_info:
                    self.index_usage[index_id] += 1
        print(f"\nBatch execution time: {batch_execution_time} s")

        self.total_recommendation_time += recommendation_time
        self.total_materialization_time += config_materialization_time
        self.total_execution_time_actual += batch_execution_time
        self.total_time_actual += recommendation_time + config_materialization_time + batch_execution_time
        self.batch_recommendation_time.append(recommendation_time)
        self.batch_materialization_time.append(config_materialization_time)
        self.batch_execution_time.append(batch_execution_time)

        print(f"\nTotal recommendation time so far --> {self.total_recommendation_time} seconds")
        print(f"Total materialization time so far --> {self.total_materialization_time} seconds")
        print(f"Total execution time so far --> {self.total_execution_time_actual} seconds")
        print(f"Total time so far --> {self.total_time_actual} seconds") 
        print(f"\nIndex usage stats:")
        for index_id in self.index_usage:
            print(f"\tIndex {index_id}: {self.index_usage[index_id]}")


    # update WFIT step for next query in workload (this is the MAIN INTERFACE for generating an index configuration recommendation)
    def process_WFIT(self, query_object, remove_stale_U=False, remove_stale_freq=1, execute=True, materialize=True, verbose=False, update_time=True):
        self.n_pos += 1        
        previous_config = list(self.M.values())
 
        # get estimated no index cost for the query
        #conn = create_connection(dbname=DBNAME)
        #self.total_no_index_cost += (hypo_query_cost(conn, query_object, [], currently_materialized_indexes=list(self.M.values())) * self.execution_cost_scaling)
        #close_connection(conn)

        # generate new partitions 
        if verbose: print(f"\nGenerating new partitions for query #{self.n_pos}")
        start_time_1 = time.time()
        new_partitions, need_to_repartition, ibg = self.choose_candidates(self.n_pos, query_object, verbose=verbose)
        end_time_1 = time.time()

        # repartition if necessary
        start_time_2 = time.time()
        if need_to_repartition:
            if verbose: print(f"Repartitioning...")
            self.repartition(new_partitions, verbose)
        end_time_2 = time.time()
        
        # analyze the query and get recommendation
        if verbose: print(f"Analyzing query...")
        start_time_3 = time.time()
        all_indexes_added, all_indexes_removed = self.analyze_query(query_object, ibg, verbose=verbose)
        end_time_3 = time.time()    

        # materialize new indexes
        if materialize:
            config_materialization_time = self.materialize_configuration(all_indexes_added, all_indexes_removed, verbose)
            if verbose: 
                print(f"\n{len(self.M)} currently materialized indexes: {[index.index_id for index in self.M.values()]}\n") 
        else:
            config_materialization_time = 0
            print(f"\n{len(self.M)} currently recommended indexes: {[index.index_id for index in self.M.values()]}\n") 

        # execute the query with the new configuration
        if execute:
            # restart the server before each query execution
            restart_postgresql()
            if verbose: print(f"Executing query...")
            conn = create_connection(dbname=DBNAME)
            execution_time, rows, table_access_info, index_access_info, bitmap_heapscan_info = execute_query(conn, query_object.query_string, with_explain=True, return_access_info=True)
            close_connection(conn)
            execution_time /= 1000
            print(f"Indexes accessed --> {list(index_access_info.keys())}")
            # update index usage stats
            for index_id in index_access_info:
                self.index_usage[index_id] += 1
        else:
            execution_time = 0

        if update_time:
            self.total_recommendation_time += end_time_3 - start_time_1     
            self.total_materialization_time += config_materialization_time
            self.total_execution_time_actual += execution_time
            self.total_time_actual += (end_time_3 - start_time_1) + config_materialization_time + execution_time   
            self.recommendation_time.append(end_time_3 - start_time_1)
            self.materialization_time.append(config_materialization_time)
            self.execution_time.append(execution_time)
        
        # remove stale indexes from U
        if remove_stale_U and (self.n_pos % remove_stale_freq == 0):
            if verbose: print(f"Removing stale indexes from U...")
            self.remove_stale_indexes_U(verbose)

        # simple recommendation, just the used indexes in the IBG root node
        self.get_simple_recommendation_ibg(ibg)

        # compute hypothetical speedup from switching to new configuration
        new_config = list(self.M.values())
        if materialize:
            materialized_indexes = self.M.values()
        else:
            materialized_indexes = []    
        conn = create_connection(dbname=DBNAME)
        speedup_wfit, query_execution_cost_wfit = hypo_query_speedup(conn, query_object, previous_config, new_config, materialized_indexes)
        self.total_cost_wfit += float(query_execution_cost_wfit) + sum([float(self.get_index_creation_cost(index)) for index in (set(new_config) - set(previous_config))])  
        # also compute speed up for the simple recommendation
        speedup_simple, query_execution_cost_simple = hypo_query_speedup(conn, query_object, previous_config, list(ibg.root.used), materialized_indexes)
        self.total_cost_simple += float(query_execution_cost_simple) + sum([float(self.get_index_creation_cost(index)) for index in (set(ibg.root.used) - set(previous_config))])
        close_connection(conn)   


        print(f"*** Hypothetical Speedup --> WFIT: {speedup_wfit}, Simple: {speedup_simple}")
        print(f"*** Hypothetical Total cost --> WFIT: {self.total_cost_wfit}, Simple: {self.total_cost_simple}, WFIT/Simple: {self.total_cost_wfit/self.total_cost_simple}")
        print(f"*** Hypothetical Total Cost No-Index --> {self.total_no_index_cost}")

        print(f"\nRecommendation time for query #{self.n_pos}: {end_time_3 - start_time_1} seconds")
        if materialize: print(f"Index materialization time for query #{self.n_pos}: {config_materialization_time} seconds")
        if execute: print(f"Execution time for query #{self.n_pos} --> {execution_time} seconds")
        print(f"\n(Partitioning: {end_time_1 - start_time_1} seconds, Repartitioning: {end_time_2 - start_time_2} seconds, Analyzing: {end_time_3 - start_time_3} seconds), Materializing config: {config_materialization_time} seconds, Executing query: {execution_time} seconds")
        
        if update_time:
            print(f"\nTotal recommendation time so far --> {self.total_recommendation_time} seconds")
            print(f"Total materialization time so far --> {self.total_materialization_time} seconds")
            print(f"Total execution time so far --> {self.total_execution_time_actual} seconds")
            print(f"Total time so far --> {self.total_time_actual} seconds")

        if execute:
            print(f"\nIndex usage stats:")
            for index_id in self.index_usage:
                print(f"\tIndex {index_id}: {self.index_usage[index_id]}")


    # Simple baseline recommendation: just the used indexes in the IBG root node, i.e. these are the indexes from 
    # the full set of candidate indexes which are used in the query plan
    def get_simple_recommendation_ibg(self, ibg):
        simple_recommendation = ibg.root.used  
        wfit_recommendation = [index.index_id for i in self.current_recommendations for index in self.current_recommendations[i]]
        #print(f"*** WFIT recommendation: {sorted(wfit_recommendation)}")
        print(f"\n*** Simple recommendation: {sorted([index.index_id for index in simple_recommendation])}\n") 


    # check for stale indexes in U and remove them
    def remove_stale_indexes_U(self, verbose):
        excess = len(self.U) - self.max_U
        if excess <= 0:
            return
        
        # get index benefit statistics
        avg_benefit = {}
        #indexes_no_data = []
        for index_id in self.U:
            # compute average benefit of the index from all stats
            """
            if len(self.idxStats[index_id]) > 0:
                avg_benefit[index_id] = sum([stat[1] for stat in self.idxStats[index_id]]) / len(self.idxStats[index_id])
            else:
                avg_benefit[index_id] = 0
                indexes_no_data.append(index_id)
            """
            if len(self.idxStats[index_id]) == 0:
                # zero current benefit if no statistics are available
                avg_benefit[index_id] = 0
            else:
                # take the maximum over all incremental average benefits (optimistic estimate)
                current_benefit = 0
                b_total = 0
                for (n, b) in self.idxStats[index_id]:
                    b_total += b 
                    # incremental average benefit of index up to query n (higher weight/smaller denominator for more recent queries)
                    benefit = b_total / (self.n_pos - n + 1)
                    current_benefit = max(current_benefit, benefit)
                avg_benefit[index_id] = current_benefit        

        # partition the indexes according to table
        table_indexes = defaultdict(list)
        for index_id in self.U:
            table_indexes[self.U[index_id].table_name].append((index_id, avg_benefit[index_id]))

        # sort indexes in each table by average benefit
        for table_id in table_indexes:
            table_indexes[table_id] = sorted(table_indexes[table_id], key=lambda x: x[1], reverse=True)   

        # print indexes in each table
        #if verbose:
        #    print(f"Indexes in each table:")
        #    for table_id in table_indexes:
        #        print(f"\tTable {table_id}: {[index[0] for index in table_indexes[table_id]]}")    

        # print(f"\nSorted indexes by table:")
        # for table_id in table_indexes:
        #     print(f"Table {table_id}: ")
        #     for index_id, benefit in table_indexes[table_id]:
        #         print(f"\t{index_id}: {benefit}")


        # collect index_id of all indexes to keep
        indexes_to_keep = self.S_0 + list(self.M.keys()) + list(self.C.keys())
        indexes_to_keep = set(indexes_to_keep)

        # for each table, keep the highest benefit index from the newest extracted indexes
        best_index_extracted = defaultdict(list)
        for table_id in table_indexes:
            if len(best_index_extracted[table_id]) == 0:
                for index_id, benefit in table_indexes[table_id]:
                    if len(self.idxStats[index_id]) == 1:
                        best_index_extracted[table_id].append(index_id)
                        break
        indexes_to_keep = indexes_to_keep.union(set(chain(*[best_index_extracted[table_id] for table_id in best_index_extracted])))                            

        if verbose:
            print(f'\nNum indexes in U: {len(self.U)}')
            print(f"Number of excess indexes: {excess}\n")
        #    print(f'Num indexes to keep: {len(indexes_to_keep)}')

        # keep iterating over tables, remove least beneficial indexes until we have self.max_U indexes and 
        # there are at least self.max_indexes_per_table indexes per table if possible
        num_removed = 0
        table_indexes_removed = defaultdict(list)
        while excess > 0:
            indexes_removed = []
            for table_id in table_indexes:
                #print(f"Table: {table_id}, num indexes: {len(table_indexes[table_id])}")
                if len(table_indexes[table_id]) > self.MAX_INDEXES_PER_TABLE:
                    # iterate backwards from the end of ther table and removethe first index that is not in indexes_to_keep
                    for i in range(len(table_indexes[table_id])-1, self.MAX_INDEXES_PER_TABLE-1, -1):
                        index_id = table_indexes[table_id][i][0]
                        if index_id not in indexes_to_keep:
                            # remove the index from table_indexes
                            table_indexes[table_id].pop(i)
                            del self.U[index_id]
                            indexes_removed.append(index_id)
                            table_indexes_removed[table_id].append(index_id)
                            excess -= 1
                            num_removed += 1
                            break    

            #print(f"Indexes removed: {indexes_removed}, current excess: {excess}") 
            if excess == 0 or len(indexes_removed) == 0:
                break                        

        if verbose:
            #print(f"Average benefit of indexes:")
            #for index_id in sorted_indexes:
            #    print(f"\tIndex {index_id}: {avg_benefit[index_id]}, Stale: {index_id in stale_indexes}")
                
            print(f"\nNumber of indexes removed: {num_removed}, Number of indexes remaining: {len(self.U)}")
            #print(f"\nIndexes removed:\n")
            #for table_id in table_indexes_removed:
            #    print(f"  Table {table_id}: ")
            #    for index_id in table_indexes_removed[table_id]:
            #        print(f"      {index_id} : {avg_benefit[index_id]}")

            #print(f"Indexes in U: {self.U.keys()}")
                

    # repartition the stable partitions based on the new partitions
    def repartition(self, new_partitions, verbose):
        # all indexes recommmendations across the WFA instances from previous round
        S_curr = set(chain(*self.current_recommendations.values()))
        C = set(self.C.values()) 
        S_0 = set(self.S_0)
        if verbose: 
            print(f"Reinitializing WFA instances...")
            print(f"S0 = {[index.index_id for index in S_0]}")
            print(f"C = {[index.index_id for index in C]}\n")

        # re-initizlize WFA instances and recommendations for each new partition
        W = {}
        recommendations = {}
        for i, P in enumerate(new_partitions):
            P_minus_C = [index for index in P if index not in C]
            if verbose:
                print(f"\tNew partition # {i+1}: {[index.index_id for index in P]}")
                print(f"\tIndexes in new partition not in C: {[index.index_id for index in P_minus_C]}")
            partition_all_configs = [tuple(sorted(state, key=lambda x: x.index_id)) for state in powerset(P)]
            wf = {}
            # initialize work function values for each state
            for X in partition_all_configs:
                #if verbose: print(f"\t\tState: {tuple([index.index_id for index in X])}")
                wf_x = 0
                # iterate over old partitions and compute accumulated wf value
                for j, wf_prev in self.W.items():
                    # check if old partition overlaps with new partition
                    overlap = set(P) & set(self.stable_partitions[j])
                    #if verbose: 
                    #    print(f"\t\tOld Partition # {j+1} --> {[index.index_id for index in self.stable_partitions[j]]}")
                    #    print(f"\t\tOverlap between new partition and old partition: {[index.index_id for index in overlap]}")
    
                    if len(overlap) > 0:
                        X_intersection_Cj = [index for index in X if index in self.stable_partitions[j]] # set(X) & set(self.stable_partitions[j])
                        X_intersection_Cj_sorted = tuple(sorted(X_intersection_Cj, key=lambda x: x.index_id))  
                        wf_x += wf_prev[tuple(X_intersection_Cj_sorted)]
                        #if verbose: 
                        #    print(f"\t\t\tIntersection with old partition: {[index.index_id for index in X_intersection_Cj]}")
                        #    print(f"\t\t\tAccumulated wf value: {wf_x}")

                S_0_1 = [index for index in S_0 if index in P_minus_C] # S_0 & (set(P) - C)
                X_1 = [index for index in X if index in P_minus_C] # set(X) & (set(P) - C)
                transition_cost_term = self.compute_transition_cost(S_0_1, X_1)
                if verbose:
                    print(f"\t\tS_0_1: {[index.index_id for index in S_0_1]}, X_1: {[index.index_id for index in X_1]}")
                    print(f"\t\tTransition cost term: {transition_cost_term}")

                wf[X] = wf_x + transition_cost_term #- self.total_no_index_cost
                #if verbose: print(f"\n\t\t w[{tuple([index.index_id for index in X])}] --> {wf[X]}   ({wf_x} + {transition_cost_term} - {self.total_no_index_cost})")
                if verbose: print(f"\n\t\t w[{tuple([index.index_id for index in X])}] --> {wf[X]}   ({wf_x} + {transition_cost_term})\n")
            
            W[i] = wf
            # initialize current state/recommended configuration of the WFA instance
            P_intersect_S_curr = [index for index in P if index in S_curr] # list(set(P) & S_curr)
            recommendations[i] = P_intersect_S_curr
            if verbose: print(f"\tRecommendation for partition # {i+1}: {[index.index_id for index in recommendations[i]]}\n")

        # replace current stable partitions, WFA instances and recommendations with the new ones
        self.stable_partitions = new_partitions
        self.W = W
        self.current_recommendations = recommendations
        
        """
        if verbose: 
            print(f"Replaced stable partitions, WFA instances and recommendations with new ones")
            print(f"New WFA instances:")
            for i, wf in self.W.items():
                print(f"\tWFA Instance #{i}:")
                for X, value in wf.items():
                    print(f"\t\tState: {tuple([index.index_id for index in X])}, Work function value: {value}")
        """

        self.C = {}
        for P in self.stable_partitions:
            for index in P: 
                self.C[index.index_id] = index      


    # update WFA instance on each stable partition and get index configuration recommendation
    def analyze_query(self, query_object, ibg, verbose):
        #S_current = set(chain(*self.current_recommendations.values()))
        new_recommendations = {}
        # update WFA instance for each stable partition
        all_indexes_added = []
        all_indexes_removed = []
        for i in self.W:
            if verbose: print(f"Updating WFA instance for partition # {i+1}\n")
            self.W[i], new_recommendations[i]  = self.process_WFA(query_object, self.W[i], self.current_recommendations[i], ibg, verbose)

            # materialize new recommendation
            indexes_added = set(new_recommendations[i]) - set(self.current_recommendations[i])
            indexes_removed = set(self.current_recommendations[i]) - set(new_recommendations[i])
            if verbose: print(f"\tWFA Instance #{i}, Num States: {len(self.W[i])}, New Recommendation: {[index.index_id for index in new_recommendations[i]]} --> Indexes Added: {[index.index_id for index in indexes_added]}, Indexes Removed: {[index.index_id for index in indexes_removed]}\n")
            
            for index in indexes_added:
                self.M[index.index_id] = index
            for index in indexes_removed:
                del self.M[index.index_id]    
                
            self.current_recommendations[i] = new_recommendations[i]

            all_indexes_added += list(indexes_added)
            all_indexes_removed += list(indexes_removed)

        if verbose: print(f"\nCurrent recommendation: {list(self.M.keys())}\n")

        return all_indexes_added, all_indexes_removed


    # materialize new configuration
    def materialize_configuration(self, all_indexes_added, all_indexes_removed, verbose):
        # check if any removed indexes are pk indexes
        #pk_indexes_removed = [index for index in all_indexes_removed if index in self.pk_indexes]
        #if pk_indexes_removed:
        #    raise ValueError(f"Error: Cannot remove primary key indexes: {[index.index_id for index in pk_indexes_removed]}")
        
        # materialize new configuration
        if verbose: 
            #print(f"\nNew indexes added this round: {[index.index_id for index in all_indexes_added]}")
            #print(f"Old indexes removed this round: {[index.index_id for index in all_indexes_removed]}\n")
            print(f"Materializing new configuration...")
        
        start_time = time.time()
        conn = create_connection(dbname=DBNAME)
        bulk_drop_indexes(conn, all_indexes_removed)
        bulk_create_indexes(conn, all_indexes_added)
        close_connection(conn)
        end_time = time.time()
        creation_time = end_time - start_time
        
        return creation_time    


    # update a WFA instance for the given query    
    def process_WFA(self, query_object, wf, S_current, ibg, verbose):
        # update work function values for each state in the WFA instance
        wf_new = {}
        p = {}
        for Y in wf.keys():
            sorted_Y = tuple(sorted(Y, key=lambda x: x.index_id))
            #if verbose: print(f"\tComputing work function value for state: {tuple([index.index_id for index in sorted_Y])}, old value --> {wf[sorted_Y]}")
            # compute new work function value for state Y 
            min_wf_value = float('inf')
            wf_X = {}
            for X in wf.keys():
                sorted_X = tuple(sorted(X, key=lambda x: x.index_id))
                wf_term = wf[sorted_X]
                if self.simple_cost: 
                    query_cost_term = self.simple_cost_model.predict(list(sorted_X), verbose=False)[0] * self.execution_cost_scaling
    
                else:
                    query_cost_term = ibg.get_cost_used(list(sorted_X))[0] 
                transition_cost_term = self.compute_transition_cost(sorted_X, sorted_Y) 
                wf_value = wf_term + query_cost_term + transition_cost_term
                #if verbose: print(f'\t\tValue for X = {tuple([index.index_id for index in sorted_X])} -->  {wf_value}  ({wf_term} + {query_cost_term} + {transition_cost_term})')
                
                wf_X[sorted_X] = wf_value
                # keep track of minimum work function value for the state
                if wf_value < min_wf_value:
                    min_wf_value = wf_value

            wf_new[sorted_Y] = min_wf_value
            min_p = []
            for X in wf_X:
                if wf_X[X] == min_wf_value:
                    min_p.append(X)
            p[sorted_Y] = min_p
            #if verbose: print(f"\tUpdated value: w[{tuple([index.index_id for index in sorted_Y])}] --> {wf_new[sorted_Y]}, p: {[[index.index_id for index in indexes] for indexes in p]}")

        #if verbose:
        #    print(f"\nUpdated wf:")
        #    for state, value in wf_new.items():
        #        print(f"\tw[{tuple([index.index_id for index in state])}] = {value}")    

        # compute scores and find best state
        scores = {}
        best_score = float('inf')
        for Y in wf_new:
            score = wf_new[Y] + self.compute_transition_cost(Y, S_current)  
            scores[Y] = score
            if score < best_score: # and Y in p[Y]:
                best_score = score


        # find all states with the best score and are also in their own p set
        S_best = []
        S_best_backup = []
        for Y in wf_new:
            if scores[Y] == best_score:
                S_best_backup.append(Y)
                if Y in p[Y]:
                    S_best.append(Y)

        if S_best == []:
            raise ValueError("No best state found. There's probably a bug in the WFA update. Aborting WFIT...")
            #print(f"Warning: No best state found. There's probably a bug in the WFA update.")
            
            ##
            ## Band-aid fix: pick the state with the lowest work function value, need to find better solution
            ##
            #if () in S_best_backup:
            #    best_state = ()
            #else:
            #    best_state = min(S_best_backup, key=lambda x: wf_new[x])    
            best_state = ()

        else:
            # pick one of the best states uniformly at random
            best_state = random.choice(S_best)        

        if verbose:
            #print(f"\tAll updated Work function values for WFA instance:")
            #for Y, value in wf_new.items():
            #    print(f"\t\tstate :{tuple([index.index_id for index in Y])} , w_value: {value}, score: {scores[Y]}, p: {[[index.index_id for index in indexes] for indexes in p[Y]]}, state in p: {Y in p[Y]}") 

            #print(f"\tBest states: ")
            #for Y in S_best:
            #    print(f"\t\t{tuple([index.index_id for index in Y])} --> in p[best_state]: {Y in p[Y]}")
        
            print(f"\tBest score: {best_score}, Selected best state: {tuple([index.index_id for index in best_state])}")
            #print(f"\tBest score: {best_score}, p[best_state]: {[[index.index_id for index in indexes] for indexes in p[best_state]]}")
            
        return wf_new, best_state

    
    # compute index benefit graph for the given query and candidate indexes
    def create_IBG(self, query_object, candidate_indexes, simple_cost_model=None):
        return IBG(query_object, candidate_indexes, existing_indexes=list(self.M.values()), execution_cost_scaling=self.execution_cost_scaling, ibg_max_nodes=self.ibg_max_nodes, doi_max_nodes=self.doi_max_nodes, max_doi_iters_per_node=self.max_doi_iters_per_node, normalize_doi=self.normalize_doi, simple_cost_model=simple_cost_model)
    

    # extract candidate indexes from given query
    def extract_indexes(self, query_object, max_size_mb=4096):        
        candidate_indexes = extract_query_indexes(query_object,  self.max_key_columns, self.include_cols, self.max_include_columns, dbname=DBNAME)
        new_indexes = [index for index in candidate_indexes if index.index_id not in self.index_size]
        # get hypothetical idnex sizes
        conn = create_connection(dbname=DBNAME)
        new_index_sizes = get_hypothetical_index_sizes(conn, new_indexes)
        close_connection(conn)
        for index in new_indexes:
            self.index_size[index.index_id] = new_index_sizes[index.index_id]

        # filter out indexes that exceed the maximum size
        candidate_indexes = [index for index in candidate_indexes if self.index_size[index.index_id] <= max_size_mb]

        return candidate_indexes
        

    # generate stable partitions/sets of indexes for next query in workload
    def choose_candidates(self, n_pos, query_object, remove_stale_U, remove_stale_freq, verbose):
        
        # create a new simple cost model for the query
        if self.simple_cost:
            if WFIT._stats_cache is None:
                tables, pk_columns = get_tpch_schema()
                table_names = list(tables.keys())
                stats = {}
                estimated_rows = {}
                for table_name in table_names:
                    stats[table_name], estimated_rows[table_name] = get_table_stats(table_name, dbname=DBNAME)

                WFIT._stats_cache = stats
                WFIT._estimated_rows_cache = estimated_rows

            self.simple_cost_model = SimpleCost(query_object, WFIT._stats_cache, WFIT._estimated_rows_cache, index_scan_cost_multiplier=1.0, join_column_discount=self.join_column_discount, dbname=DBNAME) # 1.5)
        else:
            self.simple_cost_model = None

        # check if query template seen before
        if query_object.template_id in self.query_templates_seen:
            if self.enable_stable_partition_locking:
                # lock the stable partitions
                self.stable_partitions_locked = True
                if verbose: print(f"\n\n***Query template seen before, locking stable partitions...\n\n")
                new_partitions = self.stable_partitions
                need_to_repartition = False
                ibg = self.ibg_prev
                return new_partitions, need_to_repartition, ibg

        else:
            self.query_templates_seen.add(query_object.template_id)
            if verbose: print(f"\n***New query template, unlocking stable partitions...\n")
            self.stable_partitions_locked = False    
        

        # extract candidate indexes from the query
        candidate_indexes = self.extract_indexes(query_object)
        # add new candidate indexes to the list of all candidate indexes
        num_new = 0
        for index in candidate_indexes:
            if index.index_id not in self.U:
                self.U[index.index_id] = index
                num_new += 1

        if verbose: print(f"Extracted {num_new} new indexes from query.")

        #if len(self.U) > self.max_U:
        #    raise ValueError("Number of candidate indexes exceeds the maximum limit. Aborting WFIT...")

        if verbose: 
            print(f"Candidate indexes (including those currently materialized), |U| = {len(self.U)}")
            #print(f"{[index.index_id for index in self.U.values()]}")

        # TODO: need mechanism to evict indexes from U that may have gone "stale" to prevent unbounded growth of U 

        # compute index benefit graph for the query
        if verbose: print(f"Computing IBG...")
        ibg = self.create_IBG(query_object, list(self.U.values()), self.simple_cost_model)
        self.ibg_prev = ibg

        #if verbose: print(f"Candidate index sizes in Mb: {[(index.index_id,index.size) for index in self.U.values()]}")
        
        # update statistics for the candidate indexes (n_pos is the position of the query in the workload sequence)
        if verbose: print(f"Updating statistics...")
        self.update_stats(query_object, n_pos, ibg, remove_stale_U, remove_stale_freq, verbose=False)

        # non-materialized candidate indexes 
        X = [self.U[index_id] for index_id in self.U if index_id not in self.M]
        num_indexes = self.idxCnt - len(self.M)

        # determine new set of candidate indexes to monitor for upcoming workload queries
        if verbose: print(f"\nChoosing top {num_indexes} indexes from {len(X)} non-materialized candidate indexes")
        top_indexes = self.top_indexes(n_pos, X, num_indexes, verbose)
        D = self.M | top_indexes
        if verbose: print(f"\nNew set of indexes to monitor for upcoming workload, |D| = {len(D)}")

        # generate new partitions by clustering the new candidate set
        if verbose: print(f"\nChoosing new partitions...")
        new_partitions, need_to_repartition = self.choose_partition(n_pos, D, verbose)

        return new_partitions, need_to_repartition, ibg
    

    # partition the new candidate set into clusters 
    # (need to optimize this function, currently it is a naive implementation)
    # TODO: Maybe check how much the new partitions differ from the old ones before repartitioning, need to define a metric for this
    # e.g. could maybe count what fraction of the old partitions are still present in the new partitions...based on that, decide whether to repartition or not
    def choose_partition(self, N_workload, D, verbose):
        
        # compute total loss, i.e. sum of doi across indexes from pairs of partitions
        def compute_loss(P, current_doi):
            loss = 0
            for i in range(len(P)):
                for j in range(i+1, len(P)):
                    for a in P[i]:
                        for b in P[j]:
                            loss += current_doi[(a.index_id, b.index_id)]
            return loss
        
        # compute current doi values for all pairs of indexes in U
        current_doi = defaultdict(int)
        for (a_idx, b_idx) in self.intStats.keys():
            # take max over incremental averages (optimistic estimate)
            current_doi[(a_idx, b_idx)] = 0
            doi_total = 0
            for (n, doi) in self.intStats[(a_idx, b_idx)]:
                doi_total += doi
                doi_avg = doi_total / (N_workload-n+1)
                current_doi[(a_idx, b_idx)] = max(current_doi[(a_idx, b_idx)], doi_avg)
            # save symmetric doi value
            current_doi[(b_idx, a_idx)] = current_doi[(a_idx, b_idx)]    

        #if verbose:
        #    print("Current degree of interaction:")
        #    for pair, doi in current_doi.items():
        #        print(f"\tPair {pair}: {doi}")     

        # from each current stable partition, remove indexes not in D
        P = []
        for partition in self.stable_partitions:
            if len(partition) > 0:
                P.append([index for index in partition])

        # add a singleton partition containing each new index in D not in C
        for index_id, index in D.items():
            if index_id not in self.C:
                P.append([index])
        
        # set the new partition as baseline solution if feasible
        total_configurations = sum([2**len(partition) for partition in P])
        if total_configurations <= self.stateCnt:
            bestSolution = P
            bestLoss = compute_loss(P, current_doi)
        else:
            bestSolution = None
            bestLoss = float('inf')    

        # perform randomized clustering to find better solution
        for i in range(self.rand_cnt):
            # create partition of D in singletons
            P = [[index] for index in D.values()]
            partition2id = {tuple(partition):i for i, partition in enumerate(P)}
            loss_cache = {}
            
            #if verbose:
            #    print(f"Parition to id map: {partition2id}")

            # first merge all singletons, then merge pairs of partitions (randomized merge)
            # stopping condition: no feasible merge pairs left (i.e. any merge would exceed stateCnt)
            while True:
                # find all feasible merge candidates pairs (i.e. pairs with loss > 0 and 2^(|Pi|+|Pj|) <= stateCnt)
                E = []
                E1 = []

                # get loss for all pairs of partitions
                total_configurations = sum([2**len(partition) for partition in P])
                for i in range(len(P)):
                    for j in range(i+1, len(P)):
                        Pi_id = partition2id[tuple(P[i])]
                        Pj_id = partition2id[tuple(P[j])]
                        if (Pi_id, Pj_id) in loss_cache:
                            loss = loss_cache[(Pi_id, Pj_id)]
                        else:
                            loss = compute_loss([P[i], P[j]], current_doi)
                            loss_cache[(Pi_id, Pj_id)] = loss

                        # only include feasible merge pairs, i.e. a pair which can be merged without the total number of configs exceeding stateCnt
                        total_configrations_after_merge = total_configurations - 2**len(P[i]) - 2**len(P[j]) + 2**(len(P[i]) + len(P[j]))
                        if loss > 0 and total_configrations_after_merge <= self.stateCnt:
                            E.append((P[i], P[j], loss))    
                            if len(P[i]) == 1 and len(P[j]) == 1:
                                E1.append((P[i],P[j], loss))

                #if verbose:    
                    #print(f"E pairs: {[[(index.index_id for index in Pi), (index.index_id for index in Pj), loss] for (Pi, Pj, loss) in E]}")
                    #print(f"E1 pairs: {[[(index.index_id for index in Pi), (index.index_id for index in Pj), loss] for (Pi, Pj, loss) in E1]}")

                if len(E) == 0:
                    break
                
                elif len(E1) > 0:
                    # merge a random pair of singletons, sample randomly from E1 weighted by loss (i.e. high loss pairs more likely to be merged)
                    Pi, Pj, loss = random.choices(E1, weights=[loss for (Pi, Pj, loss) in E1], k=1)[0]
                    Pij_merged = Pi + Pj
                    P.remove(Pi)
                    P.remove(Pj)
                    P.append(Pij_merged) 
                    E1.remove((Pi, Pj, loss))  
                    partition2id[tuple(Pij_merged)] = len(partition2id) 
                    #if verbose: 
                    #    print(f"Merged singleton partitions {[index.index_id for index in Pi]} and {[index.index_id for index in Pj]} with loss {loss}")

                else:
                    # merge a random pair of partitions, sample randomly from E weighted by normalized loss  
                    Pi, Pj, loss = random.choices(E, weights=[loss / (2**(len(Pi) + len(Pj)) - 2**len(Pi) - 2**len(Pj)) for (Pi, Pj, loss) in E], k=1)[0]
                    Pij_merged = Pi + Pj
                    P.remove(Pi)
                    P.remove(Pj)
                    P.append(Pij_merged)   
                    E.remove((Pi, Pj, loss)) 
                    partition2id[tuple(Pij_merged)] = len(partition2id) 
                    #if verbose:
                    #    print(f"Merged partitions {[index.index_id for index in Pi]} and {[index.index_id for index in Pj]} with loss {loss}")    

            # check if the new solution is better than the current best solution
            loss = compute_loss(P, current_doi)
            if loss < bestLoss:
                bestSolution = P
                bestLoss = loss

        # check if old partitions are different from new partitions
        partition_match = []
        for partition in bestSolution:
            matched = False
            for stable_partition in self.stable_partitions:
                if sorted([index.index_id for index in partition]) == sorted([index.index_id for index in stable_partition]):
                    matched = True
                    break
            partition_match.append(matched)
        # count how many partitions are different
        num_diff = len([match for match in partition_match if not match])
        if verbose: 
            print(f"\nOld partitions:")
            for P in self.stable_partitions:
                print(f"\t{[index.index_id for index in P]}")
            print("New partitions:")
            for P in bestSolution:
                print(f"\t{[index.index_id for index in P]}")    

            print(f"\nFraction of new partitions that don't match old partitions: {num_diff}/{len(partition_match)}")
            indexes_in_new_partitions = set([index.index_id for partition in bestSolution for index in partition])
            indexes_in_old_partitions = set([index.index_id for partition in self.stable_partitions for index in partition])
            indexes_added = indexes_in_new_partitions - indexes_in_old_partitions
            indexes_removed = indexes_in_old_partitions - indexes_in_new_partitions
            print(f"{len(indexes_added)} indexes added to new partitions: {indexes_added}")
            print(f"{len(indexes_removed)} indexes removed from old partitions: {indexes_removed}\n")    

        need_to_repartition = not all(partition_match)                   

        return bestSolution, need_to_repartition


    # update candidate index statistics
    def update_stats(self, query_object, n, ibg, remove_stale_U, remove_stale_freq, verbose):
        # update index benefit statistics
        if verbose: print("Updating index benefit statistics...")
        for index in self.U.values():
            max_benefit = ibg.compute_max_benefit(index)
            #if verbose: print(f"\tibg max benefit for index {index.index_id}: {max_benefit}")
            self.idxStats[index.index_id].append((n, max_benefit))
            #if verbose: print(f"\tIndex {index.index_id}: {self.idxStats[index.index_id]}")
            # evict old stats if the size exceeds histSize
            self.idxStats[index.index_id] = self.idxStats[index.index_id][-self.histSize:]
        
        if verbose:
            print("Index benefit statistics:")
            for index_id, stats in self.idxStats.items():
                print(f"\tIndex {index_id}: {stats}")

        # remove stale indexes from U
        if remove_stale_U and (self.n_pos % remove_stale_freq == 0):
            #if verbose: print(f"Removing stale indexes from U...")
            print(f"Removing stale indexes from U...")
            self.remove_stale_indexes_U(verbose=True)

            # recreate IBG with updated candidate indexes
            print(f"Re-computing IBG...")
            ibg = self.create_IBG(query_object, list(self.U.values()), self.simple_cost_model)
            self.ibg_prev = ibg 

        # compute all pair doi in IBG
        ibg.compute_doi()

        # update index interaction statistics
        if verbose: print("Updating index interaction statistics...")
        for (a_idx, b_idx) in ibg.doi.keys():
            d = ibg.doi[(a_idx, b_idx)]
            #if verbose: print(f"\tibg doi for pair ({a_idx}, {b_idx}) : {d}")
            if d > 0:
                self.intStats[(a_idx, b_idx)].append((n, d))
            #if verbose: print(f"\tPair ({a_idx}, {b_idx}): {self.intStats[(a_idx, b_idx)]}")
            # evict old stats if the size exceeds histSize
            self.intStats[(a_idx, b_idx)] = self.intStats[(a_idx, b_idx)][-self.histSize:]

        #if verbose:
        #    print("Index interaction statistics:")
        #    for pair, stats in self.intStats.items():
        #        print(f"\tPair {pair}: {stats}")


    # choose top num_indexes indexes from X with highest potential benefit
    def top_indexes(self, N_workload, X, num_indexes, verbose, matching_prefix_length=2, drop_zero_benefit_indexes=True):
        #if verbose:
        #    print(f"Non-materialized candidate indexes, X = {[index.index_id for index in X]}")

        # compute "current benefit" of each index in X (these are derived from statistics of observed benefits from recent queries)
        score = {}
        for index in X:
            if len(self.idxStats[index.index_id]) == 0:
                # zero current benefit if no statistics are available
                current_benefit = 0
            else:
                # take the maximum over all incremental average benefits (optimistic estimate)
                current_benefit = 0
                b_total = 0
                for (n, b) in self.idxStats[index.index_id]:
                    b_total += b 
                    # incremental average benefit of index up to query n (higher weight/smaller denominator for more recent queries)
                    benefit = b_total / (N_workload - n + 1)
                    current_benefit = max(current_benefit, benefit)

            # use current benefit to compute a score for the index
            if index.index_id in self.C:
                creation_cost_term = 0
            else:
                # if index not being monitored, then score takes a penalty for cost of creating the index
                # (unmonitored indexes are penalized so that they are only chosen if they have high potential benefit, which helps keep C stable)
                creation_cost_term = - self.get_index_creation_cost(index)

            score[index.index_id] = current_benefit + creation_cost_term
            #if verbose: print(f"\tIndex {index.index_id}: current benefit: {current_benefit}, creation penalty: {creation_cost_term}, score: {score[index.index_id]}")

        top_indexes = [index_id for index_id, s in score.items()]  
        #top_indexes = sorted(top_indexes, key=lambda x: score[x], reverse=True)[:num_indexes]
        top_indexes = sorted(top_indexes, key=lambda x: score[x], reverse=True)
        top_indexes = {index_id: self.U[index_id] for index_id in top_indexes}

        if drop_zero_benefit_indexes:
            # drop indexes with zero (or negative) benefit
            top_indexes = {index_id: index for index_id, index in top_indexes.items() if score[index_id] > 0}


        ########################################################################################################################

        # make separate lists of sorted indexes for each table
        top_indexes_table = defaultdict(list)
        for index in top_indexes.values():
            top_indexes_table[index.table_name].append(index)
            
        for table in top_indexes_table:
            top_indexes_table[table] = sorted(top_indexes_table[table], key=lambda x: score[x.index_id], reverse=True)
        
        print(f"Top indexes by table:")
        for table in top_indexes_table:
            print(f"\tTable: {table}, Indexes: ")
            for index in top_indexes_table[table]:
                print(f"\t\tIndex {index.index_id}: {score[index.index_id]}")

        # get materialized indexes by table
        materialized_indexes_table = defaultdict(list)
        for index in self.M.values():
            materialized_indexes_table[index.table_name].append(index)

        # select top indexes for each table, at most MAX_INDEXES_PER_TABLE top indexes and already materialized indexes combined per table
        top_indexes_keep = defaultdict(list)
        for table in top_indexes_table:
            num_keep = self.MAX_INDEXES_PER_TABLE - len(materialized_indexes_table[table])
            #print(f"Table: {table}, Num keep: {num_keep}")
            # add top indexes for the table one by one
            for index in top_indexes_table[table]:
                if num_keep <= 0:
                    break
                #print(f"\nConsidering candidate index: {index.index_id}")
                
                # don't add index if 
                # 1) it has a matching prefix with any of the already selected or materialized indexes 
                # 2) if all index columns other than leading columns are already covered by another index
                # 3) if it is already covered by another index


                # if len(index.index_columns) < matching_prefix_length:
                #     top_indexes_keep[table].append(index)
                #     print(f"  Adding index: {index.index_id}")
                #     num_keep -= 1
                #     continue

                index_covered = False
                for already_chosen_index in (top_indexes_keep[table] + materialized_indexes_table[table]):
                    #print(f"  Checking for matching prefix with index: {already_chosen_index.index_id}")
                    #(set(index.index_columns[1:]) == set(already_chosen_index.index_columns[1:]))

                    matching_non_leading_columns = False
                    if len(index.index_columns[1:]) > 0 and len(already_chosen_index.index_columns[1:]) > 0:
                        matching_non_leading_columns = set(index.index_columns[1:]) == set(already_chosen_index.index_columns[1:])

                    index_covered = False
                    
                    # check if the index+include columns are already covered by another chosen index's index columns
                    if set(index.index_columns + index.include_columns).issubset(set(already_chosen_index.index_columns)):
                        index_covered = True

                    # check if the index columns are the same as another chosen index's index columns and that chosen index is not materialized
                    if set(index.index_columns) == set(already_chosen_index.index_columns) and (already_chosen_index not in materialized_indexes_table[table]):
                        index_covered = True    
                    
                    # check if the index columns are the same as another chosen index's index columns and include columns are a subset of the other index's include columns    
                    if index.index_columns == already_chosen_index.index_columns and set(index.include_columns).issubset(set(already_chosen_index.include_columns)):
                        index_covered = True
                    
                    # check if the leading columns of the index are the same as another chosen index's leading columns  
                    if (index.index_columns[0] == already_chosen_index.index_columns[0]):
                        #print(f"  Found matching leading column with index: {already_chosen_index.index_id}")    
                        
                        # check if set(index+include columns) equals the set of the other index's index+include columns
                        if (set(index.index_columns + index.include_columns) == set(already_chosen_index.index_columns + already_chosen_index.include_columns)):
                            index_covered = True 

                        # check if remaining index columns + include columns are a subset of the other index's index columns + include columns    

                    # check if the index columns are a subset of another chosen index's index columns and include columns are also a subset of the other index's include columns
                    if set(index.index_columns).issubset(set(already_chosen_index.index_columns)) and set(index.include_columns).issubset(set(already_chosen_index.include_columns)):
                        index_covered = True

                    found_matching_prefix = False    
                    if (index.index_columns[:matching_prefix_length] == already_chosen_index.index_columns[:matching_prefix_length]): 
                        if (set(index.include_columns) != set(already_chosen_index.include_columns)):
                            found_matching_prefix = True 
                    
                    if found_matching_prefix or matching_non_leading_columns or index_covered:
                        index_covered = True
                        #print(f"  Already covered by index: {already_chosen_index.index_id}")
                        break

                    # check if the index columns are the same as another chosen index's index columns and the include columns are a superset of the other index's include columns, in this case, replace the other index with the current index
                    """if index.index_columns == already_chosen_index.index_columns and set(already_chosen_index.include_columns).issubset(set(index.include_columns)):
                        # check if already chosen index is materialized
                        if already_chosen_index not in materialized_indexes_table[table]:
                            top_indexes_keep[table].remove(already_chosen_index)
                            #print(f"  Replacing index: {already_chosen_index.index_id} with index: {index.index_id}")
                            break    
                        else:
                            found_matching_prefix = True
                    """

                if not index_covered:
                    top_indexes_keep[table].append(index)
                    #print(f"  Adding index: {index.index_id} -> Include columns: {index.include_columns}")
                    num_keep -= 1
                    continue

            print(f"\nSelected indexes for table: {table}: {[index.index_id for index in top_indexes_keep[table]]}")    

        # make sure the number of selected indexes per table is at most MAX_INDEXES_PER_TABLE
        for table in top_indexes_keep:
            if len(top_indexes_keep[table]) > self.MAX_INDEXES_PER_TABLE:
                raise ValueError(f"Error: Number of selected indexes for table {table} exceeds the maximum limit. Aborting WFIT...")

        top_indexes = {index.index_id: index for indexes in top_indexes_keep.values() for index in indexes}        

        if verbose:
            print(f"\n{len(top_indexes)} Top index scores:")
            for index_id in top_indexes:
                print(f"\tIndex {index_id}: {score[index_id]}")
        
            #print(f" top indexes: {[index.index_id for index in top_indexes.values()]}")

        return top_indexes    


    # return index creation cost (using estimated index size as proxy for creation cost)
    def get_index_creation_cost(self, index):
        # return estimated size of index   # (normalize by size of database)
        return self.creation_cost_fudge_factor * index.size #/ self.database_size  


    # compute transition cost between two MTS states/configurations
    def compute_transition_cost(self, S_old, S_new):
        # find out which indexes are added
        added_indexes = set(S_new) - set(S_old)
        #indexes_removed = set(S_old) - set(S_new)
        
        # compute cost of creating the added indexes
        transition_cost = sum([self.get_index_creation_cost(index) for index in added_indexes])
        # add infinite cost if any primary index is removed (we don't want to remove primary indexes)
        #if any([index.is_primary for index in indexes_removed]):
        #    transition_cost = float('inf')

        #print(f"\t\t\tComputing transition cost for state: {tuple([index.index_id for index in S_old])} --> {tuple([index.index_id for index in S_new])} = {transition_cost}")
        return transition_cost


"""
    Greedy Baseline Algorithm

    In this algorithm, for every new query, we extract all cadidate indexes $C$, then use HypoPG to find the subset $S \subseteq C$ of indexes used by the query planner and materialize the indexes in $S$ which don't exist currently.  

    Limit on maximum memory for indexes => use some form of bin packing to decide which indexes to keep in the configuration => need to allow index dropping as well as creation, maybe could drop/evict least recently used (LRU) indexes to make room for new indexes.

"""

class HypoGreedy:

    def __init__(self, config_memory_MB=2048, max_key_columns=3, include_cols=False, max_include_columns=3):
        self.currently_materialized_indexes = {}
        self.total_whatif_calls = 0
        self.total_whatif_time = 0
        # memory budget for configuration
        self.config_memory_MB = config_memory_MB
        # maximum number of key columns in an index
        self.max_key_columns = max_key_columns
        # allow include columns in indexes
        self.include_cols = include_cols
        # maximum number of include columns in an index
        self.max_include_columns = max_include_columns
        # track time 
        self.recommendation_time = []
        self.materialization_time = []
        self.execution_time = []
        self.total_recommendation_time = 0
        self.total_materialization_time = 0
        self.total_execution_time_actual = 0
        self.total_time = 0
        self.current_round = 0

        # index size cache
        self.index_size = {}

        # index_stats_cache
        self.index_stats = {}

        print(f"*** Dropping all materialized indexes...")
        conn = create_connection(dbname=DBNAME)
        drop_all_indexes(conn)
        close_connection(conn)


    # create statistics entry for a new index
    def create_index_stats(self, index_id):
        stats = {'when_selected':[], 'when_materialized':[], 'when_used':[]}
        self.index_stats[index_id] = stats


    # materialize new indexes and evict old indexes if necessary 
    def materialize_indexes(self, recommended_indexes):
        # materialize new recommendation
        indexes_added = [index for index in recommended_indexes if index.index_id not in self.currently_materialized_indexes]
        # compute size of currently materialized indexes (use hypothetical sizes)
        current_size = sum([self.index_size[index.index_id] for index in self.currently_materialized_indexes.values()])
        print(f"Size of current configuration: {current_size} MB, Space needed for new indexes: {sum([self.index_size[index.index_id] for index in indexes_added])} MB")
        # remove least recently used indexes to make space for new indexes if necessary
        indexes_removed = []
        while current_size + sum([self.index_size[index.index_id] for index in indexes_added]) > self.config_memory_MB:
            if current_size == 0:
                # this means the new indexes are too big to fit in the memory budget, need to pack a subset of them
                print(f"New indexes are too big to fit in the memory budget, packing a subset of them...")
                # sort the new indexes in descending order of size
                indexes_added = sorted(indexes_added, key=lambda x: self.index_size[x.index_id], reverse=True)
                # pack the indexes until the memory budget is reached
                packed_indexes = []
                for index in indexes_added:
                    if current_size + self.index_size[index.index_id] <= self.config_memory_MB:
                        packed_indexes.append(index)
                        current_size += self.index_size[index.index_id]    
                indexes_added = packed_indexes
                break
            
            # find the least recently selected index
            lru_index = min(self.currently_materialized_indexes.values(), key=lambda x: self.index_stats[x.index_id]['when_selected'][-1])
            print(f"Evicting index {lru_index.index_id} to make space for new indexes")
            # remove the index from the materialized indexes
            indexes_removed.append(lru_index)
            del self.currently_materialized_indexes[lru_index.index_id]
            # update the current size
            current_size -= self.index_size[lru_index.index_id]

        for index in indexes_added:
            self.currently_materialized_indexes[index.index_id] = index

        print(f"New indexes added this round: {[index.index_id for index in indexes_added]}")
        print(f"Old indexes removed this round: {[index.index_id for index in indexes_removed]}")

        # materialize new configuration
        conn = create_connection(dbname=DBNAME)
        bulk_drop_indexes(conn, indexes_removed)
        close_connection(conn)
        print(f"Materializing new indexes...")
        start_time = time.time()
        conn = create_connection(dbname=DBNAME)
        bulk_create_indexes(conn, indexes_added)
        close_connection(conn)
        end_time = time.time()
        creation_time = end_time - start_time

        # update index usage stats
        for index in indexes_added:
            self.index_stats[index.index_id]['when_materialized'].append(self.current_round)


        print(f"Currently materialized indexes: {list(self.currently_materialized_indexes.keys())}")

        return creation_time


    # extract candidate indexes from given query
    def extract_indexes(self, query_object):        
        candidate_indexes = extract_query_indexes(query_object,  self.max_key_columns, self.include_cols, self.max_include_columns, dbname=DBNAME)
        new_indexes = [index for index in candidate_indexes if index.index_id not in self.index_size]
        # get hypothetical idnex sizes
        conn = create_connection(dbname=DBNAME)
        new_index_sizes = get_hypothetical_index_sizes(conn, new_indexes)
        close_connection(conn)
        for index in new_indexes:
            self.index_size[index.index_id] = new_index_sizes[index.index_id]

        # create index stats for new indexes
        for index in new_indexes:
            self.create_index_stats(index.index_id)
        # update index stats for candidate indexes
        for index in candidate_indexes:
            self.index_stats[index.index_id]['when_selected'].append(self.current_round)    

        return candidate_indexes


    # get hypothetical cost and used indexes for the query in the given configuration, recommend the used indexes
    def get_recommendation(self, query_string, indexes):
        start_time = time.time()
        conn = create_connection(dbname=DBNAME)
        # hide existing indexes
        bulk_hide_indexes(conn, list(self.currently_materialized_indexes.values()))
        # create hypothetical indexes
        hypo_indexes = bulk_create_hypothetical_indexes(conn, indexes)
        # map oid to index object
        oid2index = {}
        for i in range(len(hypo_indexes)):
            oid2index[hypo_indexes[i]] = indexes[i]
        # get cost and used indexes
        cost, indexes_used = get_query_cost_estimate_hypo_indexes(conn, query_string, show_plan=False)
        # map used index oids to index objects
        used = [oid2index[oid] for oid, scan_type, scan_cost in indexes_used]
        # drop hypothetical indexes
        bulk_drop_hypothetical_indexes(conn)
        # unhide existing indexes
        bulk_unhide_indexes(conn, list(self.currently_materialized_indexes.values()))
        close_connection(conn)
        end_time = time.time()

        # Store the result in the class-level cache
        #self._class_cache[indexes_tuple] = (cost, used)
        self.total_whatif_calls += 1
        self.total_whatif_time += end_time - start_time

        # remove duplicates
        used = list({index.index_id:index for index in used}.values())

        print(f"Recommended indexes: {[index.index_id for index in used]}")

        return used
    
    # execute the query
    def execute(self, query_object):
        # restart the server before each query execution
        restart_postgresql()
        conn = create_connection(dbname=DBNAME)
        execution_time, rows, table_access_info, index_access_info, bitmap_heapscan_info = execute_query(conn, query_object.query_string, with_explain=True, return_access_info=True)
        close_connection(conn)

        # update index usage stats
        for index_id in index_access_info:
            self.index_stats[index_id]['when_used'].append(self.current_round)    

        print(f"Execution time: {execution_time/1000} s")
        print(f"Indexes accessed --> {list(index_access_info.keys())}")
        return execution_time


    # process the query using greedy algorithm
    def process_greedy(self, query_object):
        self.current_round += 1
        #print(f"Round# {self.n_rounds}")

        start_time = time.time()
        # extract candidate indexes
        print("Extracting candidate indexes...")
        candidate_indexes = self.extract_indexes(query_object)

        # get index recommendation
        print("Getting index recommendation...")
        recommended_indexes = self.get_recommendation(query_object.query_string, candidate_indexes)
        end_time = time.time()
        recommendation_time = end_time - start_time
        self.recommendation_time.append(recommendation_time)

        # materialize indexes
        print("Materializing indexes...")
        materialization_time = self.materialize_indexes(recommended_indexes)
        self.materialization_time.append(materialization_time)

        # execute query
        print("Executing query...")
        execution_time = self.execute(query_object)
        self.execution_time.append(execution_time)

        self.total_recommendation_time += recommendation_time
        self.total_materialization_time += materialization_time
        self.total_execution_time_actual += execution_time
        self.total_time += recommendation_time + materialization_time + (execution_time/100)

        print(f"\nTotal recommendation time so far: {self.total_recommendation_time} s")
        print(f"Total materialization time so far: {self.total_materialization_time} s")
        print(f"Total execution time so far: {self.total_execution_time_actual/1000} s")
        print(f"Total time spent so far: {self.total_time} s")
        

""" 
    No Index Baseline
"""

def execute_workload_noIndex(workload, drop_indexes=False, restart_server=False, clear_cache=False):
    if drop_indexes:
        print(f"*** Dropping all existing indexes...")
        # drop all existing indexes
        conn = create_connection(dbname=DBNAME)
        drop_all_indexes(conn)
        close_connection(conn)

    print(f"Executing workload without any indexes...")
    # execute workload without any indexes
    total_time = 0
    query_execution_time = []
    for i, query_object in enumerate(workload):
        if restart_server:
            # restart the server before each query execution
            restart_postgresql(clear_cache=clear_cache)
        
        print(f"\nExecuting query# {i+1} --> template_id: {query_object.template_id} ...")
        conn = create_connection(dbname=DBNAME)
        execution_time, rows, table_access_info, index_access_info, bitmap_heapscan_info = execute_query(conn, query_object.query_string, with_explain=True, return_access_info=True)
        close_connection(conn)
        execution_time /= 1000
        query_execution_time.append(execution_time)
        print(f"\tExecution_time: {execution_time} s, Indexes accessed: {list(index_access_info.keys())}\n")
        total_time += execution_time

    print(f"Total execution time for workload without any indexes: {total_time} seconds")
    
    return total_time, query_execution_time    











