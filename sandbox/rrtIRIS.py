import numpy as np
from numpy.core.fromnumeric import argmin
from pydrake.all import (MathematicalProgram, Variable, HPolytope, le) 

class IrisNode:
    def __init__(self, region, ellipse, parent = None):
        self.ellipse = ellipse
        self.region = region
        self.parent = parent
        self.children = []
        self.id = None

class RRTIRIS:
    def __init__(self, 
                 start, 
                 goal, 
                 limits, 
                 iris_handle,
                 col_func_handle, 
                 init_goal_sample_rate = 0.05,
                 goal_sample_rate_scaler = 0.1,
                 verbose = False,
                 plotcallback = None,
                 sample_collision_free = False
                 ):


        #col_check(pos) == True -> in collision!
        self.in_collision = col_func_handle
        self.check_col = True
        
        if self.check_col and self.in_collision(start):
            print("[RRT ERROR] Start position is in collision")
            raise NotImplementedError

        if self.check_col and self.in_collision(goal):
            print("[RRT ERROR] Goal position is in collision")
            raise NotImplementedError

        region, ellipse = iris_handle(seed = start)
        root = IrisNode(region, ellipse)
        root.id = 0
        self.nodes = [root]
        self.node_pos = [self.nodes[0].pos] 
        self.start = start
        self.goal = goal
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
    
        self.init_goal_sample_rate = init_goal_sample_rate
        self.goal_sample_rate = init_goal_sample_rate
        self.goal_sample_rate_scaler = goal_sample_rate_scaler
        self.verbose = verbose
        self.plotcallback = plotcallback
        self.do_plot = True if self.plotcallback is not None else False
        self.sample_collision_free =sample_collision_free
        
        self.dim = len(start)
        self.distance_to_go = 1e9

       

    def get_closest_node_in_regions(self, pos):
        problem = self.build_closest_points_problem(pos)
        sol = problem.solve()
        points = sol.x.reshape(-1,self.dim)
        dists = np.linalg.norm(pos.reshape(-1, self.dim) - points, axis = 1)
        idx = np.argmin(dists)
        closest_point_in_region = points[idx, :] 
        min_dist = dists[idx]
        return 

    def sample_node_pos(self, collision_free = True, MAXIT = 1e4):
        rand = np.random.rand()
        if rand < self.goal_sample_rate:
            return self.goal
        
        rand = np.random.rand(self.dim)
        pos_samp = self.min_pos + rand*self.min_max_diff 
        if self.sample_collision_free:
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
            print("[RRT ERROR] Could not find collision free point in MAXIT")
            raise NotImplementedError
        return pos_samp

    def check_region(self, region, ellipse):
        return True

    def run(self, n_it):

        for it in range(n_it):
            if it%3 == 0:
                print(it)
            pos_samp = self.sample_node_pos()
            nearest_id, seed_point = self.get_closest_node_in_regions(pos_samp) 
            parent_node = self.nodes[nearest_id]
            region, ellipse = self.iris_handle(seed = seed_point)
            if self.check_region(region, ellipse):
                child_node = IrisNode(region, ellipse)
                child_node.id = it + 1
                self.nodes.append(child_node)
                child_node.parent = parent_node
                
                dist_to_target = np.linalg.norm(self.goal - seed_point)
            else:
                print('[RRT IRIS] Region check failed')
            

            if self.do_plot:
                self.plotcallback(parent_node, child_node, pos_samp, it)

            if dist_to_target<self.distance_to_go:
                self.distance_to_go = dist_to_target
                self.closest_id = child_node.id
                self.goal_sample_rate = np.clip(0.8 - self.distance_to_go*self.goal_sample_rate_scaler, a_min = self.init_goal_sample_rate, a_max = 1)
                if self.verbose:
                    print("[RRT IRIS] it: {iter} distance to target: {dist: .3f} goalsample prob: {prob: .3f}".format(iter =it, dist = self.distance_to_go, prob = self.goal_sample_rate))

            if self.distance_to_go< self.extend_step_size:
                break

        #walk back through tree to get closest node    
        path = []
        current_node = self.nodes[self.closest_id]
        while current_node.parent is not None:
            current_node = current_node.parent
            path.append(current_node.pos)
        
        if self.distance_to_go <= self.extend_step_size:
            print('[RRT IRIS] Collision free path found in ', it,' regions')
            return True, path[::-1]
        else:
            print('[RRT IRIS] Could not find path in ', it,' regions')
            return False, path[::-1]
       
    def build_closest_points_problem(self, sample):

        num_regions = len(self.nodes) 
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dim*num_regions, 'x')  
        cost = x - np.tile(sample, (num_regions, 1))
        cost = cost@cost.T
        for idx in range(num_regions):
            A = self.nodes[idx].region.A()
            b = self.nodes[idx].region.b()
            prog.AddConstraint(le(A@x[idx*self.dim:(idx + 1)*self.dim]-b, 0))
        
        return prog