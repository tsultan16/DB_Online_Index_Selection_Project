""" 

    Index Benefit Graph implementation (Schnaitter PhD Thesis, 2011) 

"""



class Node:
    def __init__(self, id, indexes):
        self.id = id
        self.indexes = indexes
        self.children = []
        self.parents = []
        self.built = False
        self.cost = None
        self.used = None


# class for creating and storing the IBG
class IBG:
    def __init__(self, query_object, C):
        self.q = query_object
        self.C = C
        print(f"Number of candidate indexes: {len(self.C)}")
        #print(f"Candidate indexes: {self.C}")
        
        # map index_id to integer
        self.idx2id = {index.index_id:i for i, index in enumerate(self.C)}
        self.idx2index = {index.index_id:index for index in self.C}
        
        # create a hash table for keeping track of all created nodes
        self.nodes = {}
        # create a root node
        self.root = Node(self.get_configuration_id(self.C), self.C)
        self.nodes[self.root.id] = self.root
        print(f"Created root node with id: {self.root.id}")
        # start the IBG construction
        print("Constructing IBG...")
        self.construct_ibg(self.root)
        # compute all pair degree of interaction
        print(f"Computing all pair degree of interaction...")
        self.doi = self.compute_all_pair_doi()


    # assign unique string id to a configuration
    def get_configuration_id(self, indexes):
        # get sorted list of integer ids
        ids = sorted([self.idx2id[idx.index_id] for idx in indexes])
        return "_".join([str(i) for i in ids])
    

    # obtain cost and used indexes for a given configuration
    def _get_cost_used(self, indexes):
        conn = create_connection()
        # create hypothetical indexes
        hypo_indexes = bulk_create_hypothetical_indexes(conn, indexes)
        # map oid to index object
        oid2index = {}
        for i in range(len(hypo_indexes)):
            oid2index[hypo_indexes[i][0]] = indexes[i]
        # get cost and used indexes
        cost, indexes_used = get_query_cost_estimate_hypo_indexes(conn, self.q.query_string, show_plan=False)
        # map used index oids to index objects
        used = [oid2index[oid] for oid,scan_type,scan_cost in indexes_used]
        # drop hypothetical indexes
        bulk_drop_hypothetical_indexes(conn)
        close_connection(conn)   
        return cost, used

    # recursive IBG construction algorithm
    def construct_ibg(self, Y):
        if Y.built:
            return 
        
        # obtain query optimizers cost and used indexes
        cost, used = self._get_cost_used(Y.indexes)
        Y.cost = cost
        Y.used = used
        Y.built = True
        
        #print(f"Creating node for configuration: {[idx.index_id for idx in Y.indexes]}")
        #print(f"Cost: {cost}, Used indexes:")
        #for idx in used:
        #    print(f"{idx}")

        # create children
        for a in Y.used:
            # create a new configuration with index a removed from Y
            X_indexes = [index for index in Y.indexes if index != a]
            X_id = self.get_configuration_id(X_indexes)
            
            # if X is not in the hash table, create a new node and recursively build it
            if X_id not in self.nodes:
                X = Node(X_id, X_indexes)
                X.parents.append(Y)
                self.nodes[X_id] = X
                Y.children.append(X)
                self.construct_ibg(X)

            else:
                X = self.nodes[X_id]
                Y.children.append(X)
                X.parents.append(Y)


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
        while (len(Y_indexes - X_indexes) != 0) or (len(Y.children) > 0):               
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
    def compute_doi_configuration(self, a, b, X):
        # X must not contain a or b
        if a in X or b in X:
            raise ValueError("a or b is already in X")

        doi = abs(self.compute_benefit(a, X) - self.compute_benefit(a, X + [b]))
        doi /= self.get_cost_used(X + [a,b])[0]   
        return doi
   
    
    # computes the degree of interaction between all pairs of indexes (a,b) in candidate set C
    # Note: doi is symmetric, i.e. doi(a,b) = doi(b,a)
    def compute_all_pair_doi(self):
        # hash table for storing doi values
        doi = {}
        # intialize doi values to zero
        for i in range(len(self.C)):
            for j in range(i+1, len(self.C)):
                doi[(self.C[i].index_id, self.C[j].index_id)] = 0

        S_idxs = set([index.index_id for index in self.C])

        # iterate over each IBG node
        for Y in self.nodes.values():
            # remove Y.used from S
            Y_idxs = set([index.index_id for index in Y.indexes])
            S_Y = list(S_idxs - Y_idxs)
            # iterate over all pairs of indexes in S_Y
            for i in range(len(S_Y)):
                for j in range(i+1, len(S_Y)):
                    a_idx = S_Y[i]
                    b_idx = S_Y[j]
                     
                    # find Ya covering node in IBG
                    Ya = (Y_idxs - {a_idx, b_idx}) | {a_idx}
                    Ya = [self.idx2index[idx] for idx in Ya]
                    Ya = self.find_covering_node(Ya).indexes
                    # find Yab covering node in IBG
                    Yab = (Y_idxs - {a_idx, b_idx}) | {a_idx, b_idx}
                    Yab = [self.idx2index[idx] for idx in Yab]
                    Yab = self.find_covering_node(Yab).indexes

                    used_Y = self.get_cost_used(Y.indexes)[1]
                    used_Ya = self.get_cost_used(Ya)[1]
                    used_Yab = self.get_cost_used(Yab)[1]
                    
                    Uab = set([index.index_id for index in used_Y]) | set([index.index_id for index in used_Ya]) | set([index.index_id for index in used_Yab]) 
                    # find Yb_minus covering node in IBG 
                    Yb_minus = list((Uab - {a_idx, b_idx}) | {b_idx})
                    Yb_minus = [self.idx2index[idx] for idx in Yb_minus]
                    Yb_minus = self.find_covering_node(Yb_minus).indexes
                    # find Yb_plus covering node in IBG
                    Yb_plus = list((Y_idxs - {a_idx, b_idx}) | {b_idx})
                    Yb_plus = [self.idx2index[idx] for idx in Yb_plus]
                    Yb_plus = self.find_covering_node(Yb_plus).indexes

                    # generate quadruples
                    quadruples = [(Y.indexes, Ya, Yb_minus, Yab), (Y.indexes, Ya, Yb_plus, Yab)]

                    # compute doi using the quadruples
                    for Y_indexes, Ya_indexes, Yb_indexes, Yab_indexes in quadruples:
                        cost_Y = self.get_cost_used(Y_indexes)[0]
                        cost_Ya = self.get_cost_used(Ya_indexes)[0]
                        cost_Yb = self.get_cost_used(Yb_indexes)[0]
                        cost_Yab = self.get_cost_used(Yab_indexes)[0]
                        d = abs(cost_Y - cost_Ya - cost_Yb + cost_Yab) / cost_Yab
                        if (a_idx, b_idx) in doi:
                            doi[(a_idx,b_idx)] = max(doi[(a_idx,b_idx)], d)
                        elif (b_idx, a_idx) in doi:
                            doi[(b_idx,a_idx)] = max(doi[(b_idx,a_idx)], d)
                        else:
                            raise ValueError("Invalid pair of indexes")    
                            
        
        return doi


    # get precomputed degree of interaction between a pair of indexes
    def get_doi_pair(self, a, b):
        if (a.index_id, b.index_id) in self.doi:
            return self.doi[(a.index_id, b.index_id)]
        elif (b.index_id, a.index_id) in self.doi:
            return self.doi[(b.index_id, a.index_id)]
        else:
            raise ValueError("Invalid pair of indexes")


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