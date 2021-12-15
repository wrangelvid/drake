import numpy as np


def convert_t_to_q(t):
    q =np.arctan2(2*t/(1+t**2), (1-t**2)/(1+t**2))
    return q[:,[0,2,1]]

def convert_q_to_t(q):
    return np.tan(np.divide(q,2))[:,[0,2,1]]