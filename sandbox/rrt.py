import numpy as np

class Node:
    def __init__(self, pos, parent = None):
        self.pos = pos
        self.parent = parent
        self.children = []
        self.id = None


class RRT:
    def __init__(self, 
                 start, 
                 goal, 
                 limits, 
                 col_func_handle = None, 
                 max_extend_length = 1e-1,
                 extend_steps = 1e-2, 
                 goal_sample_rate = 0.05
                 ):

        #col_check(pos) == True -> in collision!
        self.in_collision = col_func_handle
        self.check_col = False if self.in_collision == None else True
        
        if self.check_col and self.check_col(start):
            print("[RRT ERROR] Start position is in collision")
            raise NotImplementedError

        if self.check_col and self.check_col(goal):
            print("[RRT ERROR] Goal position is in collision")
            raise NotImplementedError

        root = Node(start)
        root.id = 0
        self.nodes = [root]
        self.node_pos = [self.nodes[0].pos] 
        self.start = start
        self.goal = goal
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
        self.goal_sample_rate = goal_sample_rate
        self.max_extend_length = max_extend_length
        self.extend_step_size = extend_steps
        self.max_extend_steps =  int(max_extend_length/self.extend_step_size)
        self.dim = len(start)
        self.distance_to_go = 1e9

    def get_closest_node(self, pos):
        dists = np.linalg.norm(pos - self.node_pos)
        id_min = np.argmin(dists)
        return id_min

    def sample_node_pos(self, collision_free = True, MAXIT = 1e4):
        rand = np.random.rand()
        if rand < self.goal_sample_rate:
            return self.goal
        
        rand = np.random.rand(self.dim)
        pos_samp = self.min_pos + rand*self.min_max_diff 
        good_sample = not self.in_collision(pos_samp) if collision_free else True
            
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

    def grow_naive(self, parent, pos_samp):
        dir = pos_samp - parent.pos
        dir = np.divide(dir, np.linalg.norm(dir) + 1e-9)

        steps = 1
        child_pos = parent.pos

        if self.check_col:
            while steps < self.max_extend_steps:
                check = child_pos + dir * steps * self.extend_step_size
                if self.in_collision(check):
                   # if is in collision return last position that wasnt 
                   break  
                child_pos = check
                steps +=1
            child = Node(child_pos, parent)
            parent.children.append(child)
            return child
            
        else:
            child = Node(child_pos + self.max_extend_length*dir, parent)
            parent.children.append(child)
            return child

    def run(self, n_it):
        for it in range(n_it):
            pos_samp = self.sample_node_pos()
            nearest_id = self.get_closest_node(pos_samp) 
            parent_node = self.nodes[nearest_id]
            child_node = self.grow_naive(parent_node, pos_samp)
            child_node.id = it
            self.nodes.append(child_node)
            self.node_pos.append(child_node.pos)
            dist_to_target = np.linalg.norm(self.goal - self.node_pos[-1])
            if dist_to_target<self.distance_to_go:
                self.distance_to_go = dist_to_target
                print('distance to target:', self.distance_to_go)
        
        #walk back through tree to get closest node