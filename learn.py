import numpy as np
from collections import deque

# Learning parameters
GAMMA = 0.95
THETA = #???
ALPHA = 0.1

# Environment parameters
GRID_HEIGHT = 100       #Define square grid
GRID_WIDTH = 100
ORIENT_VAL_NUM = 360    #One for each degree
ACTION_NUM = 6          #Number of different actions = #{forward, back, left, right, acw, cw}

TRANSLATION_DIST = 2

#Table for the q values
#Dimensions are in order (x position, y position, orientation, action)
#Actions are ordered as indicated on line 6
q_table = np.zeros(shape=(GRID_WIDTH, GRID_HEIGHT, ORIENT_VAL_NUM, ACTION_NUM), dtype=float)
#Priority queue - implemented using collections library deque
p_queue = []

initial_state = [0,0,0] # states defined as tuples, 
                        # [x position, y position, degrees anticlockwise from x-axis]

ITER_NUM = 1e6          #Number of prioritised sweeping iterations


S = initial_state
for(i in range(ITER_NUM)):
    A = greedy_pol(S)
    R, Sprime = env_response(S,A)
    P = math.abs(R+GAMMA*max(q_table[Sprime[0],Sprime[1],Sprime[2]])-q_table[S[0],S[1],S[2],A)
    if P > THETA:
        p_queue.append(


def greedy_pol(S):
    return np.argmax(qtable[S[0],S[1],S[2]])

def env_response(S,A):
    reward_func = [-10,-10,-100,-100,-1,-1]
    #TODO
    action_func = [ lambda orient: ,    # move forward by TRANSLATION_DIST 
                    lambda orient: ,    # move back by TRANSLATION_DIST
                    lambda orient: ,    # move left by TRANSLATION_DIST
                    lambda orient: ,    # move right by TRANSLATION_DIST
                    lambda orient: ,    # rotate 1 degree anti clockwise
                    lambda orient: ]    # rotate 1 degree clockwise

    R = reward_func[A]
    Sprime = snap_to_grid(S + action_func[A])

    return R, Sprime
