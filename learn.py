import numpy as np

GRID_HEIGHT = 100       #Define square grid
GRID_WIDTH = 100
ORIENT_VAL_NUM = 360    #One for each degree
ACTION_NUM = 6          #Number of different actions = #{up, down, left, right, acw, cw}

#Table for the q values
q_table = np.zeros(shape=(GRID_HEIGHT, GRID_WIDTH, ORIENT_VAL_NUM, ACTION_NUM), dtype=float)
#Priority queue - unknown length so using a list
p_queue = []

ITER_NUM = 1e6          #Number of prioritised sweeping iterations

for(i in range(ITER_NUM)):
    
