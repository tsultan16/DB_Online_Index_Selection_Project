"""
    WFIT Implementation
"""


import sys
import os

# Add Postgres directory to sys.path
module_path = os.path.abspath('/home/tanzid/Code/DBMS/PostgreSQL')
if module_path not in sys.path:
    sys.path.append(module_path)

from pg_utils import *
from ssb_qgen_class import *

from collections import defaultdict, deque
import time
import random
from more_itertools import powerset
from itertools import chain
from tqdm import tqdm
import concurrent.futures
#from functools import lru_cache



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
    #_class_cache = {}

    def __init__(self, query_object, C, existing_indexes=[], ibg_max_nodes=100, doi_max_nodes=50, max_doi_iters_per_node=100, normalize_doi=True, ):
        self.q = query_object
        self.C = C
        self.existing_indexes = existing_indexes # indexes currently materialized in the database
        self.normalize_doi = normalize_doi
        print(f"Number of candidate indexes: {len(self.C)}")
        #print(f"Candidate indexes: {self.C}")
        
        # create a connection session to the database
        self.conn = create_connection()
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
        print("Constructing IBG...")
        self.construct_ibg(self.root, max_nodes=ibg_max_nodes)
        print(f"Number of nodes in IBG: {len(self.nodes)}, Total number of what-if calls: {self.total_whatif_calls}, Time spent on what-if calls: {self.total_whatif_time}")
        # compute all pair degree of interaction
        print(f"Computing all pair degree of interaction...")
        start_time = time.time()
        #self.doi = self.compute_all_pair_doi()
        self.doi = self.compute_all_pair_doi_parallel(num_workers=4, max_nodes=doi_max_nodes, max_iters_per_node=max_doi_iters_per_node)
        #self.doi = self.compute_all_pair_doi_simple()
        #self.doi = self.compute_all_pair_doi_naive(num_samples=256)
        #print(f"All pair doi:")
        #for key, value in self.doi.items():
        #    print(f"{key}: {value}")
        
        end_time = time.time()
        print(f"Time spent on computing all pair degree of interaction: {end_time - start_time}")

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
    def _get_cost_used(self, indexes):
        # Convert indexes to a tuple to make it hashable
        #indexes_tuple = tuple(sorted(indexes, key=lambda x: x.index_id))
        # Check if the result is already in the class-level cache
        #if indexes_tuple in self._class_cache:
        #    return self._class_cache[indexes_tuple]
        
        start_time = time.time()
        if self.conn is None:
            conn = create_connection()
        else:
            conn = self.conn    
        # create hypothetical indexes
        hypo_indexes = bulk_create_hypothetical_indexes(conn, indexes)
        # map oid to index object
        oid2index = {}
        for i in range(len(hypo_indexes)):
            oid2index[hypo_indexes[i]] = indexes[i]
        # get cost and used indexes
        cost, indexes_used = get_query_cost_estimate_hypo_indexes(conn, self.q.query_string, show_plan=False)
        # map used index oids to index objects
        used = [oid2index[oid] for oid, scan_type, scan_cost in indexes_used]
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


        return cost, used

    # Ensure the indexes parameter is hashable
    def _cached_get_cost_used(self, indexes):
        return self._get_cost_used(tuple(indexes))

    
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
                        print(f"Creating node # {self.node_count}", end="\r")
         
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
    
                if iter_count >= max_iters_per_node:
                    break
            if iter_count >= max_iters_per_node:
                break

    return doi_chunk




"""
    WFIT Implementation
"""

class WFIT:

    def __init__(self, S_0=[], max_key_columns=None, include_cols=False, max_U=30, ibg_max_nodes=100, doi_max_nodes=50, max_doi_iters_per_node=100, idxCnt=25, stateCnt=500, histSize=100, rand_cnt=100, creation_cost_fudge_factor=2048):
        # initial set of materialzed indexes
        self.S_0 = S_0
        # maximum number of key columns in an index
        self.max_key_columns = max_key_columns
        # allow include columns in indexes
        self.include_cols = include_cols
        # maximum number of candidate indexes for IBG 
        self.max_U = max_U
        # maximum number of nodes in IBG
        self.ibg_max_nodes = ibg_max_nodes
        # maximum number of nodes in DOI computation
        self.doi_max_nodes = doi_max_nodes
        # maximum number of iterations per node in DOI computation
        self.max_doi_iters_per_node = max_doi_iters_per_node
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
        # growing list of candidate indexes (initially contains S_0)
        self.U = {index.index_id:index for index in S_0}
        # index benefit and interaction statistics
        self.idxStats = defaultdict(list)
        self.intStats = defaultdict(list)
        # list of currently monitored indexes
        self.C = {index.index_id:index for index in S_0} 
        # list of currently materialized indexes
        self.M = {index.index_id:index for index in S_0}  
        # initialize stable partitions (each partition is a singleton set of indexes from S_0)
        self.stable_partitions = [[index] for index in S_0] if S_0 else [[]]
        # keep track of candidate index sizes
        self.index_size = {}
        self.n_pos = 0

        print(f"##################################################################")
        # initialize work function instance for each stable partition
        self.W = self.initilize_WFA(self.stable_partitions)
        # initialize current recommendations for each stable partition
        self.current_recommendations = {i:indexes for i, indexes in enumerate(self.stable_partitions)}


        print(f"Initial set of materialized indexes: {[index.index_id for index in S_0]}")
        print(f"Stable partitions: {[[index.index_id for index in P] for P in self.stable_partitions]}")
        print(f"Initial work function instances: ")
        for i, wf in self.W.items():
            print(f"\tWFA Instance #{i}: {wf}")

        print(f"\nMaximum number of candidate indexes tracked: {idxCnt}")
        print(f"Maximum number of MTS states/configurations: {stateCnt}")
        print(f"Maximum number of historical index statistics kept: {histSize}")
        print(f"Number of randomized clustering iterations: {rand_cnt}")

        # bulk drop all materialized indexes
        print(f"*** Dropping all materialized indexes...")
        conn = create_connection()
        drop_all_indexes(conn)
        close_connection(conn)
        print(f"##################################################################\n")

        # set random seed
        random.seed(1234)
        # track time 
        self.recommendation_time = []
        self.materialization_time = []
        self.execution_time = []
        self.total_recommendation_time = 0
        self.total_materialization_time = 0
        self.total_execution_time_actual = 0
        self.total_time_actual = 0
        self.total_cost_wfit = 0
        self.total_cost_simple = 0
        self.total_no_index_cost = 0

        # constants
        self.MAX_INDEXES_PER_TABLE = 5


    # initialize a WFA instance for each stable partition
    def initilize_WFA(self, stable_partitions):
        print(f"Initializing WFA instances for {len(stable_partitions)} stable partitions...")
        W = {}
        for i, P in enumerate(stable_partitions):
            # initialize all MTS states, i.e. power set of indexes in the partition
            states = [tuple(sorted(state, key=lambda x: x.index_id)) for state in powerset(P)]
            # initialize work function instance for the partition
            W[i] = {tuple(X):self.compute_transition_cost(self.S_0, X) for X in states}    

        for i in W:
            print(f"WFA instance #{i}: {W[i]}")

        return W


    # update WFIT step for next query in workload (this is the MAIN INTERFACE for generating an index configuration recommendation)
    def process_WFIT(self, query_object, remove_stale_U=False, remove_stale_freq=1, execute=True, materialize=True, verbose=False):
        self.n_pos += 1        
        previous_config = list(self.M.values())
 
        # get estimated no index cost for the query
        conn = create_connection()
        self.total_no_index_cost += hypo_query_cost(conn, query_object, [], currently_materialized_indexes=list(self.M.values()))
        close_connection(conn)

        # generate new partitions 
        if verbose: print(f"Generating new partitions for query #{self.n_pos}")
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
        all_indexes_added, all_indexes_removed = self.analyze_query(query_object, ibg, verbose=False)
        end_time_3 = time.time()    

        # materialize new indexes
        if materialize:
            config_materialization_time = self.materialize_configuration(all_indexes_added, all_indexes_removed, verbose)
        else:
            config_materialization_time = 0

        if verbose: 
            print(f"{len(self.M)} currently materialized indexes: {[index.index_id for index in self.M.values()]}") 

        # execute the query with the new configuration
        if execute:
            # restart the server before each query execution
            restart_postgresql()
            if verbose: print(f"Executing query...")
            conn = create_connection()
            execution_time, rows, table_access_info, index_access_info, bitmap_heapscan_info = execute_query(conn, query_object.query_string, with_explain=True, return_access_info=True)
            close_connection(conn)
            execution_time /= 1000
            print(f"Indexes accessed --> {list(index_access_info.keys())}")
        else:
            execution_time = 0

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
        conn = create_connection()
        speedup_wfit, query_execution_cost_wfit = hypo_query_speedup(conn, query_object, previous_config, new_config, materialized_indexes)
        self.total_cost_wfit += float(query_execution_cost_wfit) + sum([float(self.get_index_creation_cost(index)) for index in (set(new_config) - set(previous_config))])  
        # also compute speed up for the simple recommendation
        speedup_simple, query_execution_cost_simple = hypo_query_speedup(conn, query_object, previous_config, list(ibg.root.used), materialized_indexes)
        self.total_cost_simple += float(query_execution_cost_simple) + sum([float(self.get_index_creation_cost(index)) for index in (set(ibg.root.used) - set(previous_config))])
        close_connection(conn)   


        print(f"*** Hypothetical Speedup --> WFIT: {speedup_wfit}, Simple: {speedup_simple}")
        print(f"*** Hypothetical Total cost --> WFIT: {self.total_cost_wfit}, Simple: {self.total_cost_simple}, WFIT/Simple: {self.total_cost_wfit/self.total_cost_simple}")
        print(f"*** Hypothetical Total Cost No-Index --> {self.total_no_index_cost}")

        print(f"Total recommendation time taken for query #{self.n_pos}: {end_time_3 - start_time_1} seconds")
        print(f"Actual execution time for query #{self.n_pos} --> {execution_time} seconds")
        print(f"\nTotal recommendation time so far --> {self.total_recommendation_time} seconds")
        print(f"Total materialization time so far --> {self.total_materialization_time} seconds")
        print(f"Total execution time so far --> {self.total_execution_time_actual} seconds")
        print(f"Total time so far --> {self.total_time_actual} seconds")
        print(f"(Partitioning: {end_time_1 - start_time_1} seconds, Repartitioning: {end_time_2 - start_time_2} seconds, Analyzing: {end_time_3 - start_time_3} seconds), Materializing config: {config_materialization_time} seconds, Executing query: {execution_time} seconds")


    # Simple baseline recommendation: just the used indexes in the IBG root node, i.e. these are the indexes from 
    # the full set of candidate indexes which are used in the query plan
    def get_simple_recommendation_ibg(self, ibg):
        simple_recommendation = ibg.root.used  
        wfit_recommendation = [index.index_id for i in self.current_recommendations for index in self.current_recommendations[i]]
        print(f"*** WFIT recommendation: {sorted(wfit_recommendation)}")
        print(f"*** Simple recommendation: {sorted([index.index_id for index in simple_recommendation])}") 


    # check for stale indexes in U and remove them
    def remove_stale_indexes_U(self, verbose):
        # find out which indexes have loweest benefit statistics
        avg_benefit = {}
        for index_id in self.U:
            # compute average benefit of the index from all stats
            avg_benefit[index_id] = sum([stat[1] for stat in self.idxStats[index_id]]) / len(self.idxStats[index_id])

        # sort indexes by average benefit
        sorted_indexes = sorted(avg_benefit, key=avg_benefit.get, reverse=True)

        # mark all indexes with zero benefit and not in M and S_0 as stale
        stale_indexes = set()
        for index_id in sorted_indexes:
            if avg_benefit[index_id] == 0 and index_id not in self.M and index_id not in self.S_0:
                stale_indexes.add(index_id)

        # remove stale indexes from U
        print(f"Number of indexes in U: {len(self.U)}")
        num_removed = 0
        for index_id in stale_indexes:
            #if verbose: print(f"Removing stale index: {index_id}")
            del self.U[index_id]
            #if verbose: print(f"Number of indexes in U after removal: {len(self.U)}")
            num_removed += 1

        # keep at most self.max_U of highest benefit indexes in U, make sure to keep all indexes in S_0 and M
        if self.max_U is not None and len(self.U) > self.max_U:
            for index_id in sorted_indexes[self.max_U:]:
                if index_id not in self.M and index_id not in self.S_0 and index_id in self.U:
                    del self.U[index_id]
                    num_removed += 1

        # remove stale indexes from stable partitions and C (not sure if this is necessary...)
        
        if verbose:
            #print(f"Average benefit of indexes:")
            #for index_id in sorted_indexes:
            #    print(f"\tIndex {index_id}: {avg_benefit[index_id]}, Stale: {index_id in stale_indexes}")
                
            print(f"Number of indexes removed: {num_removed}, Number of indexes remaining: {len(self.U)}")
            #print(f"Indexes in U: {self.U.keys()}")
                

    # repartition the stable partitions based on the new partitions
    def repartition(self, new_partitions, verbose):
        # all indexes recommmendations across the WFA instances from previous round
        S_curr = set(chain(*self.current_recommendations.values()))
        C = set(self.C.values()) 
        S_0 = set(self.S_0)

        # compute L2-norm of the work function across all partitions
        #l2_norm_wf_old = 0
        #for i, wf in self.W.items():
        #    l2_norm_wf_old += sum([wf[X]**2 for X in wf])

        
        # re-initizlize WFA instances and recommendations for each new partition
        if verbose: print(f"Reinitializing WFA instances...")
        W = {}
        recommendations = {}
        for i, P in enumerate(new_partitions):
            partition_all_configs = [tuple(sorted(state, key=lambda x: x.index_id)) for state in powerset(P)]
            wf = {}
            # initialize work function values for each state
            #print(f"\tNew partition # {i}")
            for X in partition_all_configs:
                wf_x = 0
                for j, wf_prev in self.W.items(): 
                    wf_x += wf_prev[tuple(sorted(set(X) & set(self.stable_partitions[j]), key=lambda x: x.index_id))]
                
                transition_cost_term = self.compute_transition_cost(S_0 & (set(P) - C), set(X) - C)
                wf[X] = wf_x + transition_cost_term - self.total_no_index_cost
                #print(f"\t\t w[{tuple([index.index_id for index in X])}] --> {wf[X]}   ({wf_x} + {transition_cost_term})")
            
            W[i] = wf
            # initialize current state/recommended configuration of the WFA instance
            recommendations[i] = list(set(P) & S_curr)

        """
        # compute l2 norm of the work function across all partitions
        l2_norm_wf_new = 0
        for i, wf in W.items():
            l2_norm_wf_new += sum([wf[X]**2 for X in wf])
        # rescale work function values to maintain the same l2 norm (otherwise wf values will keep increasing
        # due to the summation terms in repartitioning)
        for i, wf in W.items():
            for X in wf:
                wf[X] = wf[X] * (l2_norm_wf_old / l2_norm_wf_new)    
        """

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
        S_current = set(chain(*self.current_recommendations.values()))
        new_recommendations = {}
        # update WFA instance for each stable partition
        all_indexes_added = []
        all_indexes_removed = []
        for i in self.W:
            if verbose: print(f"Updating WFA instance: {i}")
            #self.W[i], new_recommendations[i]  = self.process_WFA(self.W[i], self.current_recommendations[i], ibg, verbose)
            self.W[i], new_recommendations[i]  = self.process_WFA(self.W[i], S_current, ibg, verbose)

            # materialize new recommendation
            indexes_added = set(new_recommendations[i]) - set(self.current_recommendations[i])
            indexes_removed = set(self.current_recommendations[i]) - set(new_recommendations[i])
            if verbose: print(f"\tWFA Instance #{i}, Num States: {len(self.W[i])}, New Recommendation: {[index.index_id for index in new_recommendations[i]]} --> Indexes Added: {[index.index_id for index in indexes_added]}, Indexes Removed: {[index.index_id for index in indexes_removed]}")
            
            for index in indexes_added:
                self.M[index.index_id] = index
            for index in indexes_removed:
                del self.M[index.index_id]    
                
            self.current_recommendations[i] = new_recommendations[i]

            all_indexes_added += list(indexes_added)
            all_indexes_removed += list(indexes_removed)

        return all_indexes_added, all_indexes_removed


    # materialize new configuration
    def materialize_configuration(self, all_indexes_added, all_indexes_removed, verbose):
        # materialize new configuration
        if verbose: 
            print(f"New indexes added this round: {[index.index_id for index in all_indexes_added]}")
            print(f"Old indexes removed this round: {[index.index_id for index in all_indexes_removed]}")

            print(f"Materializing new configuration...")
        start_time = time.time()
        conn = create_connection()
        bulk_drop_indexes(conn, all_indexes_removed)
        bulk_create_indexes(conn, all_indexes_added)
        close_connection(conn)
        end_time = time.time()
        creation_time = end_time - start_time
        
        return creation_time    


    # update a WFA instance for the given query    
    def process_WFA(self, wf, S_current, ibg, verbose):
        # update work function values for each state in the WFA instance
        wf_new = {}
        p = {}
        for Y in wf.keys():
            sorted_Y = tuple(sorted(Y, key=lambda x: x.index_id))
            #print(f"\tComputing work function value for state: {tuple([index.index_id for index in sorted_Y])}, old value --> {wf[sorted_Y]}")
            # compute new work function value for state Y 
            min_wf_value = float('inf')
            wf_X = {}
            for X in wf.keys():
                sorted_X = tuple(sorted(X, key=lambda x: x.index_id))
                wf_term = wf[sorted_X]
                query_cost_term = ibg.get_cost_used(list(sorted_X))[0]
                transition_cost_term = self.compute_transition_cost(sorted_X, sorted_Y) 
                wf_value = wf_term + query_cost_term + transition_cost_term
                #print(f'\t\tValue for X = {tuple([index.index_id for index in sorted_X])} -->  {wf_value}  ({wf_term} + {query_cost_term} + {transition_cost_term})')
                
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
            #print(f"\tUpdated value: w[{tuple([index.index_id for index in sorted_Y])}] --> {wf_new[sorted_Y]}, p: {[[index.index_id for index in indexes] for indexes in p]}")

        # compute scores and find best state
        best_score = float('inf')
        best_state = None  
        for Y in wf_new:
            sorted_Y = tuple(sorted(Y, key=lambda x: x.index_id))
            score = wf_new[sorted_Y] + self.compute_transition_cost(sorted_Y, S_current)  
            if score < best_score and sorted_Y in p[sorted_Y]:
                best_score = score
                best_state = sorted_Y  #min_p

        if verbose:
            #print(f"\tAll updated Work function values for WFA instance:")
            #for Y, value in wf_new.items():
            #    print(f"\t\tstate :{tuple([index.index_id for index in Y])} , w_value: {value}, p: {[[index.index_id for index in indexes] for indexes in p]}, score: {scores[Y]}")

            print(f"\tBest state: {tuple([index.index_id for index in best_state])}, Best score: {best_score}")
        
        return wf_new, best_state

    
    # compute index benefit graph for the given query and candidate indexes
    def compute_IBG(self, query_object, candidate_indexes):
        return IBG(query_object, candidate_indexes, existing_indexes=list(self.M.values()), ibg_max_nodes=self.ibg_max_nodes, doi_max_nodes=self.doi_max_nodes, max_doi_iters_per_node=self.max_doi_iters_per_node)
    

    # extract candidate indexes from given query
    def extract_indexes(self, query_object, max_size_mb=4096):        
        candidate_indexes = extract_query_indexes(query_object,  self.max_key_columns, self.include_cols)
        new_indexes = [index for index in candidate_indexes if index.index_id not in self.index_size]
        # get hypothetical idnex sizes
        conn = create_connection()
        new_index_sizes = get_hypothetical_index_sizes(conn, new_indexes)
        close_connection(conn)
        for index in new_indexes:
            self.index_size[index.index_id] = new_index_sizes[index.index_id]

        # filter out indexes that exceed the maximum size
        candidate_indexes = [index for index in candidate_indexes if self.index_size[index.index_id] <= max_size_mb]

        return candidate_indexes
        

    # generate stable partitions/sets of indexes for next query in workload
    def choose_candidates(self, n_pos, query_object, verbose):
        # extract candidate indexes from the query
        candidate_indexes = self.extract_indexes(query_object)
        # add new candidate indexes to the list of all candidate indexes
        num_new = 0
        for index in candidate_indexes:
            if index.index_id not in self.U:
                self.U[index.index_id] = index
                num_new += 1

        #if len(self.U) > self.max_U:
        #    raise ValueError("Number of candidate indexes exceeds the maximum limit. Aborting WFIT...")

        if verbose: 
            print(f"Extracted {num_new} new indexes from query.")
            print(f"Candidate indexes (including those currently materialized), |U| = {len(self.U)}")
            #print(f"{[index.index_id for index in self.U.values()]}")

        # TODO: need mechanism to evict indexes from U that may have gone "stale" to prevent unbounded growth of U

        
        # compute index benefit graph for the query
        if verbose: print(f"Computing IBG...")
        ibg = self.compute_IBG(query_object, list(self.U.values()))

        #if verbose: print(f"Candidate index sizes in Mb: {[(index.index_id,index.size) for index in self.U.values()]}")
        
        # update statistics for the candidate indexes (n_pos is the position of the query in the workload sequence)
        if verbose: print(f"Updating statistics...")
        self.update_stats(n_pos, ibg, verbose=False)

        # non-materialized candidate indexes 
        X = [self.U[index_id] for index_id in self.U if index_id not in self.M]
        num_indexes = self.idxCnt - len(self.M)

        # determine new set of candidate indexes to monitor for upcoming workload queries
        if verbose: print(f"Choosing top {num_indexes} indexes from {len(X)} non-materialized candidate indexes")
        top_indexes = self.top_indexes(n_pos, X, num_indexes, verbose)
        D = self.M | top_indexes
        if verbose: print(f"New set of indexes to monitor for upcoming workload, |D| = {len(D)}")

        # generate new partitions by clustering the new candidate set
        if verbose: print(f"Choosing new partitions...")
        new_partitions, need_to_repartition = self.choose_partition(n_pos, D, verbose)
        if verbose:
            print(f"Old partitions:")
            for P in self.stable_partitions:
                print(f"\t{[index.index_id for index in P]}")
            print("New partitions:")
            for P in new_partitions:
                print(f"\t{[index.index_id for index in P]}")    

        return new_partitions, need_to_repartition, ibg
    

    # partition the new candidate set into clusters 
    # (need to optimize this function, currently it is a naive implementation)
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
            #P.append([index for index in partition if index.index_id in D])
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
        need_to_repartition = False
        if bestSolution != self.stable_partitions:
            need_to_repartition = True

        return bestSolution, need_to_repartition


    # update candidate index statistics
    def update_stats(self, n, ibg, verbose):
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

        if verbose:
            print("Index interaction statistics:")
            for pair, stats in self.intStats.items():
                print(f"\tPair {pair}: {stats}")


    # choose top num_indexes indexes from X with highest potential benefit
    def top_indexes(self, N_workload, X, num_indexes, verbose):
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
                # if index already being monitored, then score is just current benefit
                score[index.index_id] = current_benefit
            else:
                # if index not being monitored, then score is current benefit minus cost of creating the index
                # (unmonitored indexes are penalized so that they are only chosen if they have high potential benefit, which helps keep C stable)
                score[index.index_id] = current_benefit - self.get_index_creation_cost(index)

        top_indexes = [index_id for index_id, s in score.items()]  
        #top_indexes = sorted(top_indexes, key=lambda x: score[x], reverse=True)#[:num_indexes]
        top_indexes = sorted(top_indexes, key=lambda x: score[x], reverse=True)[:num_indexes]
        top_indexes = {index_id: self.U[index_id] for index_id in top_indexes}

        """  
        # for each table, keep at most MAX_INDEXES_PER_TABLE indexes
        table_indexes = defaultdict(list)
        for index in top_indexes.values():
            table_indexes[index.table_name].append(index)
        for table in table_indexes:
            table_indexes[table] = sorted(table_indexes[table], key=lambda x: score[x.index_id], reverse=True)[:self.MAX_INDEXES_PER_TABLE]

        top_indexes = {index.index_id: index for indexes in table_indexes.values() for index in indexes}
        
        # if there are more than num_indexes indexe in top_indexes, then shave off the excess indexes
        #if len(top_indexes) > num_indexes:
        #    top_indexes = {index.index_id: index for index in top_indexes[:num_indexes]}    

        # if there are less than num_indexes indexes in top_indexes, then add indexes from the sorted top_indexes list to make up the difference
        if len(top_indexes) < num_indexes:
            for index in sorted(score, key=lambda x: score[x], reverse=True):
                if index not in top_indexes:
                    top_indexes[index] = self.U[index]
                if len(top_indexes) == num_indexes:
                    break    
    """

        if verbose:
            print(f"{len(top_indexes)} top indexes: {[index.index_id for index in top_indexes.values()]}")

        return top_indexes    


    # return index creation cost (using estimated index size as proxy for cost)
    def get_index_creation_cost(self, index):
        # return estimated size of index
        return index.size * self.creation_cost_fudge_factor  


    # compute transition cost between two MTS states/configurations
    def compute_transition_cost(self, S_old, S_new):
        # find out which indexes are added
        added_indexes = set(S_new) - set(S_old)
        
        # compute cost of creating the added indexes
        transition_cost = sum([self.get_index_creation_cost(index) for index in added_indexes])
        #print(f"\t\t\tComputing transition cost for state: {tuple([index.index_id for index in S_old])} --> {tuple([index.index_id for index in S_new])} = {transition_cost}")
        return transition_cost




"""
    Greedy Baseline Algorithm

    In this algorithm, for every new query, we extract all cadidate indexes $C$, then use HypoPG to find the subset $S \subseteq C$ of indexes used by the query planner and materialize the indexes in $S$ which don't exist currently.  

    TODO: Put limit on maximum memory for indexes => use some form of bin packing to decide which indexes to keep in the configuration => need to allow index dropping as well as creation, maybe could drop/evict least recently used (LRU) indexes to make room for new indexes.

"""

class HypoGreedy:

    def __init__(self, config_memory_MB=2048, max_key_columns=3, include_cols=False):
        self.currently_materialized_indexes = {}
        self.total_whatif_calls = 0
        self.total_whatif_time = 0
        # memory budget for configuration
        self.config_memory_MB = config_memory_MB
        # maximum number of key columns in an index
        self.max_key_columns = max_key_columns
        # allow include columns in indexes
        self.include_cols = include_cols
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
        conn = create_connection()
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
        conn = create_connection()
        bulk_drop_indexes(conn, indexes_removed)
        close_connection(conn)
        print(f"Materializing new indexes...")
        start_time = time.time()
        conn = create_connection()
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
        candidate_indexes = extract_query_indexes(query_object,  self.max_key_columns, self.include_cols)
        new_indexes = [index for index in candidate_indexes if index.index_id not in self.index_size]
        # get hypothetical idnex sizes
        conn = create_connection()
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
        conn = create_connection()
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
        conn = create_connection()
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
        








