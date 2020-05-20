# TODO fix q_table invalid actions hack

import math
import numpy as np
import render
from operator import add
from queue import PriorityQueue

class Environment:
    def __init__(self, polygons, ship_length, grid_dims):
        self.grid_dims = grid_dims
        for poly in polygons:
            assert(self._poly_in_grid(poly, grid_dims))
        self.polygons = polygons
        self.ship_length = ship_length

    def ship_ends(self, state):
        return [[state[0]+math.cos(math.radians(state[2]))*0.5*self.ship_length, 
                 state[1]-math.sin(math.radians(state[2]))*0.5*self.ship_length],
                [state[0]-math.cos(math.radians(state[2]))*0.5*self.ship_length, 
                 state[1]+math.sin(math.radians(state[2]))*0.5*self.ship_length]]

    def _poly_in_grid(self, poly, grid_dims):
        b = True
        for vertex in poly.v:
            if vertex[0] >= grid_dims[0] or vertex[0] < 0 or vertex[1] >= grid_dims[1] or vertex[1] < 0:
                b = False
        return b

class Polygon:
    def __init__(self, v):
        self.v = v
        self.e = [[v[i],v[i+1]] for i in range(len(v)-1)]
        self.e.append([v[len(v)-1],v[0]])
        self.outside = [min([vert[0] for vert in v])-1, min([vert[1] for vert in v])-1]

### Learning parameters ###
# GONNA NEED SOME TESTING
GAMMA = 0.95
THETA = 1.0
ALPHA = 0.1
ITER_NUM = int(1e3)          #Number of prioritised sweeping iterations
TERM_REWARD = 1000

### Environment parameters ### 
SHIP_LENGTH = 5

#Define square grid

#BIG GRID#
#GRID_WIDTH = 50
#GRID_HEIGHT = 50
#POLYGONS = [Polygon([[0,20],[25,20],[25,25],[0,25]]),
#            Polygon([[20,25],[25,25],[25,40],[20,40]])]
#ENV = Environment(POLYGONS, SHIP_LENGTH, (GRID_WIDTH, GRID_HEIGHT))
#initial_state = np.array([5,5,0]) # states defined as tuples, 
                        # [x position, y position, degrees anticlockwise from x-axis]
#termination_state = np.array([19,33,270])

#SMALL GRID#
GRID_WIDTH = 20
GRID_HEIGHT = 20
POLYGONS = [Polygon([[0,10],[10,10],[10,11],[0,11]])]
ENV = Environment(POLYGONS, SHIP_LENGTH, (GRID_WIDTH, GRID_HEIGHT))
initial_state = np.array([5,5,0]) # states defined as tuples, 
                        # [x position, y position, degrees anticlockwise from x-axis]
termination_state = np.array([5,15,180])


ORIENT_VAL_NUM = 360    #One for each degree
ACTION_NUM = 6          #Number of different actions = #{forward, back, left, right, acw, cw}

TRANSLATION_DIST = 2


### Render params ###
WIN_WIDTH = 500
WIN_HEIGHT= 500
FRAME_TIME_MS = 30

### INITIALISING TABLE ###
q_table = np.zeros(shape=(GRID_WIDTH, GRID_HEIGHT, ORIENT_VAL_NUM, ACTION_NUM), dtype=float)
#Priority queue - implemented using collections library deque
p_queue = PriorityQueue()

def main():
    print("initialising_q_table")
    new_env=True
    q_table = init_q_table(new_env)
    print("q_table initialised")
    print("learning q")
    learn_q()
    states, actions, rewards = run_pi()
    px_scale = min(int(WIN_WIDTH/GRID_WIDTH), int(WIN_HEIGHT/GRID_HEIGHT))
    episode = render.Episode(states, ENV, px_scale, [WIN_WIDTH,WIN_HEIGHT])
    episode.animate(FRAME_TIME_MS)

#Table for the q values
#Dimensions are in order (x position, y position, orientation, action)
#Actions are ordered as indicated on line 6
#Note that here we are initialising our q_table with an optimistic expectation for most state-action pairs, since most incur negative rewards
def init_q_table(new_env):
    import os.path

    if os.path.isfile("q_table.npy") and not new_env:
        f = open("q_table.npy", "r")
        table = np.load(f)
        q_table = table
    else:
        q_table = np.zeros(shape=(GRID_WIDTH, GRID_HEIGHT, ORIENT_VAL_NUM, ACTION_NUM), dtype=float)
        f = open("q_table.npy", "w")
        for i in range(GRID_WIDTH):
            print((i/GRID_WIDTH)*100,"%")
            for j in range(GRID_HEIGHT):
                for r in range(ORIENT_VAL_NUM):
                    for a in range(ACTION_NUM):
                        if(not is_valid(env_response([i,j,r],a)[1], ENV)):
                            q_table[i,j,r,a] = -np.inf
        #TODO FIXME
        #np.save(f, q_table)
    f.close()
    return q_table

#Note: Ensure that actions returned are legal
def greedy_pol(S):
    return np.argmax(q_table[S[0],S[1],S[2]])

#TODO implement
def epsilon_greedy_pol(S):
    #REMEMBER when choosing action, must check that it is legal
    return -1

def env_response(S,A):
    reward_func_list = [-10,-10,-100,-100,-1,-1]
    def reward_func(Sprime, A):
        if np.all(np.equal(Sprime, termination_state)):
            return TERM_REWARD 
        else:
            return reward_func_list[A]

    action_func = [ lambda orient: TRANSLATION_DIST*np.array([math.cos(math.radians(orient)), math.sin(math.radians(orient)),0]),   # move forward by TRANSLATION_DIST 
                    lambda orient: -TRANSLATION_DIST*np.array([math.cos(math.radians(orient)), math.sin(math.radians(orient)),0]),  # move back by TRANSLATION_DIST
                    lambda orient: TRANSLATION_DIST*np.array([math.cos(math.radians(orient)+math.pi/2), math.sin(math.radians(orient)+math.pi/2),0]),    # move left by TRANSLATION_DIST
                    lambda orient: -TRANSLATION_DIST*np.array([math.cos(math.radians(orient)+math.pi/2), math.sin(math.radians(orient)+math.pi/2),0]),    # move right by TRANSLATION_DIST
                    lambda orient: np.array([0,0,1]),    # rotate 1 degree anti clockwise
                    lambda orient: np.array([0,0,-1])    # rotate 1 degree clockwise
                    ]
    Sprime = list(map(add, S, action_func[A](S[2])))
    Sprime[2] = Sprime[2]%360
    Sprime = snap_to_grid(Sprime)
    R = reward_func(Sprime, A)
    return R, Sprime

# return list of (Sprime,A) which result in S 
# REMEMBER terminal state and invalid states cannot lead to anything
def backup(S):
    # Since all actions are reversible with respect to states, 
    # the graph of states connected by actions is undirected
    reverse_action = [1,0,3,2,5,4]

    state_actions = []
    for a in range(6):
        state = env_response(S,a)
        # REMEMBER only legal, non-terminal states should go in this list
        if(is_valid(state, ENV) or state == terminal_state):
            state_actions.append((state, reverse_action[a]))
    return state_actions

def snap_to_grid(S):
    return [int(round(S[0])), int(round(S[1])), S[2]]

def learn_q():
    # LEARNING Q
    S = initial_state
    for i in range(ITER_NUM):
        print("Sweep: ", i)
        A = greedy_pol(S)   #TODO try epsilon greedy policy here
        R, Sprime = env_response(S,A)
        P = abs(R+GAMMA*max(q_table[Sprime[0],Sprime[1],Sprime[2]])-q_table[S[0],S[1],S[2],A])
        if P > THETA:
            #Since priority queue only uses integer priority, we use P multiplied to 100 and rounded
            p_queue.put(int(round(P*100)), (S,A))
            while(not p_queue.empty()):
                S, A = p_queue.pop()
                R, Sprime = env_response(S,A)
                q_table[S[0],S[1],S[2],A] = q_table[S[0],S[1],S[2],A] + ALPHA*(R+GAMMA*max(q_table[Sprime[0],Sprime[1],Sprime[2]])-q_table[S[0],S[1],S[2],A])
                for Sbar,Abar in backup(S):
                    Rbar = env_response(Sbar,Abar)
                    P = abs(Rbar+GAMMA*max(q_table[S[0],S[1],S[2]])-q_table[Sbar[0],Sbar[1],Sbar[2],Abar])
                    if P > THETA:
                        p_queue.put(int(round(P*100)), (S,A))

# TODO DECIDE WHETHER TO USE THIS, OR INITIALISE Q_TABLE
def max_valid_q(arr, S):
    maxq=-np.inf
    for a, q_val in arr:
        if is_valid(env_response([i,j,r],a)[1], ENV):
            maxq = q_val
    return maxq

def run_pi():
    states = []
    actions = []
    rewards = []
    S = initial_state
    while S!=termination_state:
        state.append(S)
        A = greedy_pol(S)
        actions.append(A)
        R, S = env_response(S,A)
        rewards.append(R)
    return states, actions, rewards


def is_valid(state, environment):
    def _intersect(l1,l2):
        return _intersect_points(l1[0],l1[1], l2[0],l2[1])

    def _ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # Return true if line segments AB and CD intersect
    def _intersect_points(A,B,C,D):
        return _ccw(A,C,D) != _ccw(B,C,D) and _ccw(A,B,C) != _ccw(A,B,D)

    valid = True
    for poly in environment.polygons:
        for edge in poly.e:
            _ship_ends = environment.ship_ends(state)
            for p in _ship_ends:
                if _intersect([p, poly.outside], edge):
                    valid = 1-valid
                if p[0] < 0 or p[0] > GRID_WIDTH or p[1] < 0 or p[1] > GRID_HEIGHT:
                    return False
            if _intersect(_ship_ends, edge):
                return False
    return valid
