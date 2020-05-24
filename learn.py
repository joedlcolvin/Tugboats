import math
import numpy as np
import render
from operator import add
from p_queue import PQueue

class Environment:
    def __init__(self, polygons, ship_length, grid_dims, term_reward, translation_dist, rewards_list, angle_inc, initial_state, final_state, num_actions):
        self.grid_dims = grid_dims
        self.polygons = []
        for poly in polygons:
            polygon = Environment.Polygon(poly)
            self.polygons.append(polygon)
            assert(self._poly_in_grid(polygon, grid_dims))
        self.ship_length = ship_length
        self.term_reward = term_reward
        self.translation_dist = translation_dist
        self.rewards_list = rewards_list
        self.angle_inc = angle_inc  #6 actions so list of 6 float rewards for the actions
        self.initial_state = initial_state
        self.final_state = final_state
        self.num_actions = num_actions
        self.num_angles = int(360/angle_inc)

    def ship_ends(self, state):
        return [[state[0]+math.cos(math.radians(state[2]*self.angle_inc))*0.5*self.ship_length, 
                 state[1]-math.sin(math.radians(state[2]*self.angle_inc))*0.5*self.ship_length],
                [state[0]-math.cos(math.radians(state[2]*self.angle_inc))*0.5*self.ship_length, 
                 state[1]+math.sin(math.radians(state[2]*self.angle_inc))*0.5*self.ship_length]]

    def _poly_in_grid(self, poly, grid_dims):
        b = True
        for vertex in poly.v:
            if vertex[0] >= grid_dims[0] or vertex[0] < 0 or vertex[1] >= grid_dims[1] or vertex[1] < 0:
                b = False
        return b

    def response(self, S, A):
        def reward_func(Sprime, A):
            if np.all(np.equal(Sprime, self.final_state)):
                return self.term_reward, True
            else:
                return self.rewards_list[A], False

        action_func = [ lambda orient: self.translation_dist*np.array(  [math.cos(math.radians(orient*self.angle_inc)), 
                                                                        math.sin(math.radians(orient*self.angle_inc)),0]),              # move forward by self.translation_dist 
                        lambda orient: -self.translation_dist*np.array( [math.cos(math.radians(orient*self.angle_inc)), 
                                                                        math.sin(math.radians(orient*self.angle_inc)),0]),              # move back by self.translation_dist
                        lambda orient: self.translation_dist*np.array(  [math.cos(math.radians(orient*self.angle_inc)+math.pi/2), 
                                                                        math.sin(math.radians(orient*self.angle_inc)+math.pi/2),0]),    # move left by self.translation_dist
                        lambda orient: -self.translation_dist*np.array( [math.cos(math.radians(orient*self.angle_inc)+math.pi/2), 
                                                                        math.sin(math.radians(orient*self.angle_inc)+math.pi/2),0]),    # move right by self.translation_dist
                        lambda orient: np.array([0,0,1]),    # rotate 1 degree anti clockwise
                        lambda orient: np.array([0,0,-1])    # rotate 1 degree clockwise
                        ]
        Sprime = list(map(add, S, action_func[A](S[2])))
        Sprime[2] = int(Sprime[2]%self.num_angles)
        Sprime = self._snap_to_grid(Sprime)
        R, termination = reward_func(Sprime, A)
        if np.all(np.equal(Sprime, self.final_state)):
            Sprime = self.initial_state
        return R, Sprime, termination

    def _snap_to_grid(self, S):
        return [int(round(S[0])), int(round(S[1])), S[2]]

    def is_valid(self, S, A):
        state = self.response(S,A)[1]

        def _intersect(l1,l2):
            return _intersect_points(l1[0],l1[1], l2[0],l2[1])

        def _ccw(A,B,C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        # Return true if line segments AB and CD intersect
        def _intersect_points(A,B,C,D):
            return _ccw(A,C,D) != _ccw(B,C,D) and _ccw(A,B,C) != _ccw(A,B,D)

        _ship_ends = self.ship_ends(state)
        for p in _ship_ends:
            if p[0] < 0 or p[0] >= self.grid_dims[0] or p[1] < 0 or p[1] >= self.grid_dims[1]:
                return False

        valid = [True, True]
        for poly in self.polygons:
            for edge in poly.e:
                for i, p in enumerate(_ship_ends):
                    if _intersect([p, poly.outside], edge):
                        valid[i] = 1-valid[i]
                if _intersect(_ship_ends, edge):
                    return False
        return valid[0] and valid[1]

    class Polygon:
        def __init__(self, v):
            self.v = v
            self.e = [[v[i],v[i+1]] for i in range(len(v)-1)]
            self.e.append([v[len(v)-1],v[0]])
            self.outside = [min([vert[0] for vert in v])-1, min([vert[1] for vert in v])-1]


### POLICY FUNCTIONS ###

#Note: Ensure that actions returned are legal
def greedy_pol(env, q_table, S):
    #return np.argmax(q_table[S[0],S[1],S[2]])
    #print("Current q table for S:", q_table[S[0],S[1],S[2]])
    return max_valid_a_over_q(env, q_table[S[0],S[1],S[2]], S)

def epsilon_greedy_pol(env, q_table, S, p_params):
    r = np.random.uniform(0.,1.)
    if(r < p_params['epsilon']):
        while True:
            a = int(np.floor(np.random.uniform(0.,1.)*env.num_actions))
            if env.is_valid(S,a):
                return a
    else:
        return greedy_pol(env, q_table, S)

def run_pi(env, q_table, p_params={'epsilon':0}):
    states = []
    actions = []
    rewards = []
    S = env.initial_state
    termination = False
    while not termination:
        states.append(S)
        A = epsilon_greedy_pol(env, q_table, S, p_params)
        actions.append(A)
        R, S, termination = env.response(S,A)
        rewards.append(R)
    return states, actions, rewards

### LEARNING FUNCTIONS ###

def priority_sweep(env, l_params, q_table):
    #Priority queue - implemented using collections library deque
    p_queue = PQueue()
    # LEARNING Q
    S = env.initial_state
    for i in range(l_params['iter_num']):
        print("State: ", S)
        A = greedy_pol(S)   #TODO try epsilon greedy policy here
        #print("A: ",A)
        R, Sprime, termination = env.response(S,A)
        #print(R, Sprime)
        P = abs(R+l_params['gamma']*max_valid_q(q_table[Sprime[0],Sprime[1],Sprime[2]],S)-q_table[S[0],S[1],S[2],A])
        #print(P)
        if P > THETA:
            #Since priority queue only uses integer priority, we use P multiplied to -100 and rounded
            p_queue.add_task((tuple(S),A), int(round(-P*100)))
            while not p_queue.empty():
                print("p_queue size: ",p_queue.qsize())
                S, A = p_queue.pop_task()
                #print("S,A:", S, A)
                R, Sprime = env.response(S,A)
                #print("R,Sprime:", R, Sprime)
                q_table[S[0],S[1],S[2],A] = q_table[S[0],S[1],S[2],A] + l_params['alpha']*(R+l_params['gamma']*max_valid_q(q_table[Sprime[0],Sprime[1],Sprime[2]],S)-q_table[S[0],S[1],S[2],A])
                print("q:",q_table[S[0],S[1],S[2],A], "\nS: ", S, "\nA", A)
                for Sbar,Abar in backup(S):
                    #print("Sbar, Abar:",Sbar, Abar)
                    Rbar = env.response(Sbar,Abar)[0]
                    #print("Rbar:", Rbar)
                    P = abs(Rbar+l_params['gamma']*max_valid_q(q_table[S[0],S[1],S[2]],S)-q_table[Sbar[0],Sbar[1],Sbar[2],Abar])
                    if P > THETA:
                        print("P:",P)
                        #print("Sbar:",Sbar)
                        p_queue.add_task((tuple(Sbar),Abar), int(round(-P*100)))
        S=Sprime

    # return list of (Sprime,A) which result in S 
    # REMEMBER terminal state and invalid states cannot lead to anything
    def backup(S):
        # Since all actions are reversible with respect to states, 
        # the graph of states connected by actions is undirected
        reverse_action = [1,0,3,2,5,4]

        state_actions = []
        for a in range(6):
            termination = env.response(S,a)[2]
            # REMEMBER only legal, non-terminal states should go in this list
            if env.is_valid(S, a) and not termination:
                state_actions.append((state, reverse_action[a]))
        #print("backups: ", state_actions)
        return state_actions

def watkins(env, l_params, p_params, q_table):
    Rtots=[]
    # LEARNING Q
    for i in range(l_params['iter_num']):
        e_table = {}
        S = env.initial_state
        Rtot = 0
        A = epsilon_greedy_pol(env, q_table, S, p_params)
        print("Iteration: ", i)
        while True:
            print(len(e_table))
            print("State: ", S)
            print("Action: ", A)
            R, Sprime, termination = env.response(S,A)
            Rtot+=R
            Aprime = epsilon_greedy_pol(env, q_table, Sprime, p_params)
            Astar = max_valid_a_over_q(env, q_table[Sprime[0],Sprime[1],Sprime[2]], S)
            if(q_table[Sprime[0],Sprime[1],Sprime[2],Astar] == q_table[Sprime[0],Sprime[1],Sprime[2], Aprime]):
                Astar = Aprime
            delta = R+l_params['gamma']*q_table[Sprime[0],Sprime[1],Sprime[2],Astar]-q_table[S[0],S[1],S[2],A]
            if (tuple(S),A) in e_table:
                e_table[(tuple(S),A)]+=1
            else:
                e_table[(tuple(S),A)] = 1
            for SA in e_table:
                S, A = SA
                q_table[S[0],S[1],S[2],A] += l_params['alpha']*delta*e_table[SA]
                if(Aprime == Astar):
                    e_table[SA] *= l_params['gamma']*lmbda
                else:
                    e_table[SA] = 0
            if termination:
                print(Rtot)
                Rtots.append(Rtot)
                break
            S=Sprime
            A=Aprime
    import matplotlib.pyplot as plt
    plt.plot(Rtots)
    plt.savefig("watkins.png")
    plt.show()

def q_learn(env, l_params, p_params, q_table):
    Rtots=[]
    # LEARNING Q
    for i in range(l_params['iter_num']):
        S = env.initial_state
        Rtot = 0
        print("Iteration: ", i)
        while True:
            A = epsilon_greedy_pol(env, q_table, S, p_params)
            print("State: ", S)
            print("Action: ", A)
            R, Sprime, termination = env.response(S,A)
            Rtot+=R
            q_table[S[0],S[1],S[2],A] = q_table[S[0],S[1],S[2],A] + l_params['alpha']*(R+l_params['gamma']*max_valid_a_over_q(env, q_table[Sprime[0],Sprime[1],Sprime[2]], S)-q_table[S[0],S[1],S[2],A])
            if termination:
                print(Rtot)
                Rtots.append(Rtot)
                break
            S=Sprime
    import matplotlib.pyplot as plt
    plt.plot(Rtots)
    plt.savefig("Qlearn.png")
    plt.show()

def sarsa(env, l_params, p_params, q_table):
    Rtots=[]
    # LEARNING Q
    for i in range(l_params['iter_num']):
        S = env.initial_state
        Rtot = 0
        A = epsilon_greedy_pol(env, q_table, S, p_params)
        print("Iteration: ", i)
        while True:
            #print("State: ", S)
            #print("Action: ", A)
            R, Sprime, termination = env.response(S,A)
            Rtot+=R
            Aprime = epsilon_greedy_pol(env, q_table, Sprime, p_params)
            q_table[S[0],S[1],S[2],A] = q_table[S[0],S[1],S[2],A] + l_params['alpha']*(R+l_params['gamma']*q_table[Sprime[0],Sprime[1],Sprime[2],Aprime]-q_table[S[0],S[1],S[2],A])
            if termination:
                print(Rtot)
                Rtots.append(Rtot)
                break
            S=Sprime
            A=Aprime
    import matplotlib.pyplot as plt
    plt.plot(Rtots)
    plt.savefig("sarsa.png")
    plt.show()

def sarsa_lambda(env, l_params, p_params, q_table):
    Rtots=[]
    # LEARNING Q
    for i in range(l_params['iter_num']):
        e_table = {}
        S = env.initial_state
        Rtot = 0
        A = epsilon_greedy_pol(env, q_table, S, p_params)
        print("Iteration: ", i)
        while True:
            print(len(e_table))
            print("State: ", S)
            print("Action: ", A)
            R, Sprime, termination = env.response(S,A)
            Rtot+=R
            Aprime = epsilon_greedy_pol(env, q_table, Sprime, p_params)
            delta = R+l_params['gamma']*q_table[Sprime[0],Sprime[1],Sprime[2],Aprime]-q_table[S[0],S[1],S[2],A]
            if (tuple(S),A) in e_table:
                e_table[(tuple(S),A)]+=1
            else:
                e_table[(tuple(S),A)] = 1
            for SA in e_table:
                S, A = SA
                q_table[S[0],S[1],S[2],A] += l_params['alpha']*delta*e_table[SA]
                e_table[SA] *= l_params['gamma']*lmbda
            #FIXME
            for SA in e_table:
                if(e_table[SA] <= e_threshold):
                    del e_table[SA]
            if termination:
                print(Rtot)
                Rtots.append(Rtot)
                break
            S=Sprime
            A=Aprime
    import matplotlib.pyplot as plt
    plt.plot(Rtots)
    plt.savefig("sarsa_lambda.png")
    plt.show()

def max_valid_q(env, q_arr, S):
    maxq=-np.inf
    for a, q_val in enumerate(q_arr):
        if env.is_valid(S, a):
            maxq = max(q_val, maxq)
    assert(maxq!=-np.inf)
    return maxq

def max_valid_a_over_q(env, q_arr, S):
    maxa=-1
    maxq=-np.inf
    for a, q_val in enumerate(q_arr):
        if env.is_valid(S, a):
            if(maxq<q_val):
                maxq=q_val
                maxa=a
    assert(maxa!=-1)
    return maxa

### MAIN FUNCTION ###

def main():
    print("defining parameters")
    # Rendering parameters
    r_params = {"win_width" : 500, "win_height" : 500, "frame_time_ms" : 50}
    # Learning parameters
    l_params = {"gamma": 1, "alpha": 0.05, "iter_num": 100}
    # Policy parameters
    p_params = {"epsilon": 0.1}

    #Environment
    print("defining environment")
    big_grid = Environment( polygons =  [[[0,20],[25,20],[25,25],[0,25]], 
                                        [[20,25],[25,25],[25,40],[20,40]]], 
                            ship_length = 5, 
                            grid_dims = [50,50], 
                            term_reward = 50, 
                            translation_dist = 2**0.5, 
                            rewards_list = [-5,-5,-10,-10,-50,-50], 
                            angle_inc = 10,
                            initial_state = np.array([5,5,0]), 
                            final_state = np.array([19,33,27]), 
                            num_actions = 6)
    small_grid = Environment(   
                            polygons = [[[0,7],[7,7],[7,10],[0,10]]],
                            ship_length = 2,
                            grid_dims = [15,15],
                            term_reward = 50,
                            translation_dist = 2**0.5,
                            rewards_list = [-5,-5,-10,-10,-50,-50], 
                            angle_inc = 10,
                            initial_state = np.array([5,5,0]), 
                            final_state = np.array([5,15,18]), 
                            num_actions = 6)
    smallest_grid = Environment(
                            polygons = [],
                            ship_length = 1,
                            grid_dims = [5,5],
                            term_reward = 20,
                            translation_dist = 2**0.5,
                            rewards_list = [-5,-5,-10,-10,-50,-50], 
                            angle_inc = 10,
                            initial_state = np.array([1,1,0]), 
                            final_state = np.array([3,3,9]), 
                            num_actions = 6)

    env = small_grid

    print("initialising q_table")
    initial_q_val = 50
    q_table = np.full((env.grid_dims[0], env.grid_dims[1], env.num_angles, env.num_actions), initial_q_val, dtype=float)
    print("learning policy with sarsa")
    sarsa(env, l_params, p_params, q_table)
    print("running policy")
    states, actions, rewards = run_pi(env, q_table, p_params)
    px_scale = min( int(r_params['win_width']/env.grid_dims[0]), 
                    int(r_params['win_height']/env.grid_dims[1]))
    print("rendering")
    episode = render.Episode(states, rewards, env, px_scale, r_params['win_width'], r_params['win_height'])
    episode.animate(r_params['frame_time_ms'])

if __name__ == "__main__":
    main()
