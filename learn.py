import numpy as np
from queue import PriorityQueue

# Learning parameters
# GONNA NEED SOME TESTING
GAMMA = 0.95
THETA = 1.0
ALPHA = 0.1
ITER_NUM = 1e6          #Number of prioritised sweeping iterations

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
p_queue = PriorityQueue()

initial_state = [0,0,0] # states defined as tuples, 
                        # [x position, y position, degrees anticlockwise from x-axis]
termination_state = [10,10,0]

def main()
    S = initial_state
    for(i in range(ITER_NUM)):
        A = greedy_pol(S)
        R, Sprime = env_response(S,A)
        P = math.abs(R+GAMMA*max(q_table[Sprime[0],Sprime[1],Sprime[2]])-q_table[S[0],S[1],S[2],A])
        if P > THETA:
            #Since priority queue only uses integer priority, we use P multiplied to 100 and rounded
            p_queue.put(math.round(P*100), (S,A))
            while(not p_queue.empty()):
                S, A = p_queue.pop()
                R, Sprime = env_response(S,A)
                q_table[S[0],S[1],S[2],A] = q_table[S[0],S[1],S[2],A] + ALPHA*(R+GAMMA*max(q_table[Sprime[0],Sprime[1],Sprime[2]])-q_table[S[0],S[1],S[2],A])
                for(Sbar,Abar in backup(S)):
                    Rbar = env_response(Sbar,Abar)
                    P = math.abs(Rbar+GAMMA*max(q_table[S[0],S[1],S[2]])-q_table[Sbar[0],Sbar[1],Sbar[2],Abar])
                    if P > THETA:
                        p_queue.put(math.round(P*100), (S,A))


def greedy_pol(S):
    return np.argmax(qtable[S[0],S[1],S[2]])

def env_response(S,A):
    reward_func_list = [-10,-10,-100,-100,-1,-1]
    reward_func = lambda Sprime,A: 0 if Sprime==termination_state else reward_func_list[A]
    action_func = [ lambda orient: ,    # move forward by TRANSLATION_DIST 
                    lambda orient: ,    # move back by TRANSLATION_DIST
                    lambda orient: ,    # move left by TRANSLATION_DIST
                    lambda orient: ,    # move right by TRANSLATION_DIST
                    lambda orient: ,    # rotate 1 degree anti clockwise
                    lambda orient: ]    # rotate 1 degree clockwise

    Sprime = snap_to_grid(S + action_func[A](S[2]))     #TODO change to use elementwise addition with mod 360 for angle
    R = reward_func(Sprime,A)

    return R, Sprime

def backup(S):
    #TODO
    return #list of (Sprime,A) which result in S
