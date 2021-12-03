import rrt
import prm
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

min_pos = np.array([-1, -1])
max_pos = np.array([1, 1])

start = np.array([-0.9, - 0.9])
goal= np.array( [0.9, 0.9] )
limits = [min_pos, max_pos]

def box_obstacle(pos, pos_box = [0, 0], width= 0.3, height = 0.3):
    width_col = pos[0] >= pos_box[0]-width/2 and pos[0] <= pos_box[0]+width/2
    height_col = pos[1] >= pos_box[1]-height/2 and pos[1] <= pos_box[1]+height/2
    return width_col and height_col 

def collision(pos):
    b1 = box_obstacle(pos, [0, -0.4], width=0.2, height=1.0)
    b2 = box_obstacle(pos, [-0.5, 0.3], width=0.6, height=0.2)
    b3 = box_obstacle(pos, [0.5, 0.3], width=0.6, height=0.2)
    return b1 or b2 or b3

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.set_xlim([min_pos[0], max_pos[0]])
ax.set_ylim([min_pos[1], max_pos[1]])
ax.scatter(start[0], start[1], c = 'r', s = 50)
ax.scatter(goal[0], goal[1], c = 'g', s = 50)
plt.show(block=False)

def plotting_fn(parent_node, child_node, pos_samp, it, ax, color):
    ax.plot([child_node.pos[0], parent_node.pos[0]], [child_node.pos[1], parent_node.pos[1]], c = color)
    ax.scatter(child_node.pos[0], child_node.pos[1], c = color, s = 5, alpha= 1.0)
           
plotting_fn_handle = partial(plotting_fn, ax=ax, color= 'k')

#RRT = rrt.RRT(start,
#              goal,
#              limits,
#              col_func_handle=collision,
#              max_extend_length=0.2,
#              extend_steps=0.01,
#              init_goal_sample_rate=0.03,
#              goal_sample_rate_scaler=0.4,
#              verbose = True, 
#              plotcallback = plotting_fn_handle)
#
#
##draw some points from initial distribution to show constraints
#for _ in range(2000):
#    pos_samp = RRT.sample_node_pos()
#    ax.scatter(pos_samp[0], pos_samp[1], c = 'k', s = 1, alpha= 0.1)
#success, path = RRT.run(1000)


def plot_prm(nodes, adjacency_list, ax, color):
    for node_idx in range(nodes.shape[0]):
        pos1 = nodes[node_idx, :]
        ax.scatter(pos1[0], pos1[1], c = color, s = 5, alpha= 1.0)
        for edge_idx in range(len(adjacency_list[node_idx])): 
            pos2 = nodes[adjacency_list[node_idx][edge_idx], :]
            ax.plot([pos1[0],pos2[0]], [pos1[1], pos2[1]], c = color)

plot_prm_hand = partial(plot_prm, ax=ax, color= 'k')


PRM = prm.PRM( 
            limits,
            num_points = 300,
            col_func_handle = collision,
            num_neighbours = 5, 
            dist_thresh = 0.3,
            num_col_checks = 10,
            verbose = True,
            plotcallback = plot_prm_hand
            )

for _ in range(2000):
    pos_samp = PRM.sample_node_pos()
    ax.scatter(pos_samp[0], pos_samp[1], c = 'r', s = 10, alpha= 0.1)

plt.show()