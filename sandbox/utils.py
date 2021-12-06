import numpy as np
import matplotlib.pyplot as plt
import meshcat

class PWLinTraj:
        def __init__(self, path, duration):
            self.path = path
            self.duration = duration
            self.num_waypoints = len(self.path)
            
        def value(self, time):
            prog_frac = np.clip(time/self.duration, a_min = 0, a_max = 0.99999)*(self.num_waypoints-1)
            prog_int = int(prog_frac)
            prog_part = prog_frac-prog_int
            wp1 = self.path[prog_int]
            wp2 = self.path[prog_int+1]
            return wp1 + prog_part*(wp2-wp1)
        
        def end_time(self,):
            return self.duration

def animate(traj, publisher, steps, runtime):
    #loop
    idx = 0
    going_fwd = True
    time_points = np.linspace(0, traj.end_time(), steps) 

    for _ in range(runtime):
        #print(idx)
        q = traj.value(time_points[idx])
        publisher(q.reshape(-1,))
        if going_fwd:
            if idx + 1 < steps:
                idx += 1
            else:
                going_fwd = False
                idx -=1
        else:
            if idx-1 >= 0:
                idx -=1
            else:
                going_fwd = True
                idx +=1

def animate_t(traj, publisher, steps, t_to_q, runtime):
    #loop
    idx = 0
    going_fwd = True
    time_points = np.linspace(0, traj.end_time(), steps) 

    for _ in range(runtime):
        #print(idx)
        q = t_to_q(traj.value(time_points[idx]).reshape(1, -1)).squeeze()
        publisher(q.reshape(-1,))
        if going_fwd:
            if idx + 1 < steps:
                idx += 1
            else:
                going_fwd = False
                idx -=1
        else:
            if idx-1 >= 0:
                idx -=1
            else:
                going_fwd = True
                idx +=1


def plot(traj, steps, runtime):
    #loop
    idx = 0
    going_fwd = True
    time_points = np.linspace(0, traj.end_time(), steps) 
    traj_list = []
    
    for _ in range(runtime):
        #print(idx)
        traj_list.append(traj.value(time_points[idx]).reshape(-1,))
        if going_fwd:
            if idx + 1 < steps:
                idx += 1
            else:
                going_fwd = False
                idx -=1
        else:
            if idx-1 >= 0:
                idx -=1
            else:
                going_fwd = True
                idx +=1
    
    traj_arr = np.array(traj_list)

    fig, ax = plt.subplots(1,1, figsize=(10,6), dpi=72*3)
    data_dims = traj_arr.shape[1]
    for joint_idx in range(data_dims):
        ax.plot(np.arange(len(traj_arr[:,joint_idx])),traj_arr[:,joint_idx], label=f'Joint {joint_idx+1}')
    ax.legend(loc='upper center', ncol=data_dims)
    ax.set_ylim([-np.pi, np.pi])
    plt.show()

def meshcat_line(x_start, x_end, width):
    x_end_shift = x_end.copy()
    x_end_shift[0:2] += width
    x_start_shift = x_start.copy()
    x_start_shift[1:2] += width
    
    points = np.array([[x_start, x_end, x_end_shift, x_start_shift]]).reshape(-1,3)
    triangles = np.array([[0,1,2],[0,2,3]]).reshape(-1,3)
    mc_geom = meshcat.geometry.TriangularMeshGeometry(points, triangles)
    return mc_geom