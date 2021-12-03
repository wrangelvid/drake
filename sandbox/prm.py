import numpy as np
from scipy.spatial import cKDTree

class Node:
    def __init__(self, pos, cost, parent = None):
        self.pos = pos
        self.parent = parent
        self.cost = cost
        self.children = []
        self.id = None


class PRM:
    def __init__(self, 
                 limits,
                 num_points,
                 col_func_handle,
                 num_neighbours = 5, 
                 dist_thresh = 0.1,
                 num_col_checks = 10,
                 verbose = False,
                 plotcallback = None
                ):

        #col_check(pos) == True -> in collision!
        self.in_collision = col_func_handle
        self.check_col = False if self.in_collision == None else True
        self.dim = len(limits[0])
        self.min_pos = limits[0]
        self.max_pos = limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
        self.num_neighbours = num_neighbours
        self.dist_thresh = dist_thresh
        self.t_check = np.linspace(0, 1, num_col_checks)
        self.plotcallback = plotcallback
        self.verbose = verbose

        #generate n samples using rejection sampling 
        nodes = []
        for idx in range(num_points):
            nodes.append(self.sample_node_pos())
            if self.verbose and idx%30 == 0:
                print('[PRM] Samples', idx)
        self.nodes = np.array(nodes)
        self.nodes_kd = cKDTree(self.nodes)   
        
        #generate edges
        self.adjacency_list = self.connect_nodes()
        if self.plotcallback:
            self.plotcallback(self.nodes, self.adjacency_list)


    def sample_node_pos(self, collision_free = True, MAXIT = 1e4):

        rand = np.random.rand(self.dim)
        pos_samp = self.min_pos + rand*self.min_max_diff 
        if self.check_col:
            good_sample = not self.in_collision(pos_samp) if collision_free else True
        else: 
            good_sample = True 

        it = 0
        while not good_sample and it < MAXIT:
            rand = np.random.rand(self.dim)
            pos_samp = self.min_pos + rand*self.min_max_diff 
            good_sample = not self.in_collision(pos_samp)
            it+=1

        if not good_sample:
            print("[PRM ERROR] Could not find collision free point in MAXIT")
            raise NotImplementedError
        return pos_samp
   
    def connect_nodes(self,):
        adjacency_list = []
        for node_idx in range(self.nodes.shape[0]):
            if self.verbose and node_idx%20 == 0:
                print('[PRM] Nodes connected:', node_idx)
            edges = []
            dists, idxs = self.nodes_kd.query(self.nodes[node_idx, :], k= self.num_neighbours, p= 2, distance_upper_bound = self.dist_thresh ) 
            #linesearch connection for collision
            for step in range(len(idxs)):
                nearest_idx =idxs[step]
                if not dists[step] == np.inf:
                    add = True
                    for t in self.t_check:
                        pos = (1-t)*self.nodes[node_idx, :] + t*self.nodes[nearest_idx, :]
                        if self.in_collision(pos):
                            add = False
                            break
                    if add:
                        edges.append(nearest_idx)

            adjacency_list.append(edges)
        return adjacency_list


    