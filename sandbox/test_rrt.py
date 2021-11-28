import rrt
import numpy as np
import matplotlib.pyplot as plt

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
    b1 = box_obstacle(pos, [0, 0], width=1.0, height=1.0)
    b2 = box_obstacle(pos, [0.5, 0.3], width=1.8, height=0.2)
    return b1 or b2

RRT = rrt.RRT(start,
              goal,
              limits,
              col_func_handle=collision,
              max_extend_length=0.2,
              extend_steps=0.01,
              init_goal_sample_rate=0.05,
              goal_sample_rate_scaler=0.1)

for _ in range(10000):
    RRT.sample_node_pos()
    
RRT.run(300)
plt.show()