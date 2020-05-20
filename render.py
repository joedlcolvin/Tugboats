import graphics
import math
import time

def render(win, environment, initial_state, px_scale):
    for domain in environment.domains:
        for i in range(len(domain.e)):
            graphics.Line(  graphics.Point(domain.e[i][0][0]*px_scale, domain.e[i][0][1]*px_scale), 
                            graphics.Point(domain.e[i][1][0]*px_scale, domain.e[i][1][1]*px_scale)).draw(win)
    ship = environment.ship_ends(initial_state)
    print(ship)
    ship_rend = graphics.Line(  graphics.Point(ship[0][0]*px_scale, ship[0][1]*px_scale), 
                                graphics.Point(ship[1][0]*px_scale, ship[1][1]*px_scale)).draw(win)
    return ship_rend

class Environment:
    def __init__(self, domains, ship_length):
        self.domains = domains
        self.ship_length = ship_length

    def ship_ends(self, state):
        #Check whether math.sin/cos does radians or degrees
        return [[state.x+math.cos(math.radians(state.r))*0.5*self.ship_length, 
                 state.y-math.sin(math.radians(state.r))*0.5*self.ship_length],
                [state.x-math.cos(math.radians(state.r))*0.5*self.ship_length, 
                 state.y+math.sin(math.radians(state.r))*0.5*self.ship_length]]

class State:
    def __init__(self, x,y,r):
        self.x = x
        self.y = y
        self.r = r

    def is_valid(self, environment):
        valid = True
        for dom in environment.domains:
            for edge in dom.e:
                for p in environment.ship_ends(self):
                    if self._intersect([p, dom.outside], edge):
                        valid = 1-valid
        return valid

    def _intersect(self, l1,l2):
        return self._intersect_points(l1[0],l1[1], l2[0],l2[1])

    def _ccw(self, A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # Return true if line segments AB and CD intersect
    def _intersect_points(self,A,B,C,D):
        return self._ccw(A,C,D) != self._ccw(B,C,D) and self._ccw(A,B,C) != self._ccw(A,B,D)

class Action:
    def __init__(self, x,y,r):
        self.x = x
        self.y = y
        self.r = r

    def is_valid(self):
        #TODO check that x,y and r define a valid action
        return

class Domain:
    def __init__(self, v):
        self.v = v
        self.e = [[v[i],v[i+1]] for i in range(len(v)-1)]
        self.e.append([v[len(v)-1],v[0]])
        self.outside = [min([vert[0] for vert in v])-1, min([vert[1] for vert in v])-1]

class Episode:
    def __init__(self, initial_state, actions, environment, px_scale, win_size):
        current_state = initial_state
        for action in actions:
            assert(action.is_valid())
            assert(current_state.is_valid(environment))
            current_state = State(  current_state.x+action.x,
                                    current_state.y+action.y,
                                    current_state.r+action.r)
        assert(current_state.is_valid(environment))

        self.initial_state = initial_state
        self.environment = environment
        self.actions = actions
        self.px_scale = px_scale
        self.win_size

    def animate(self, delay):
        win = GraphWin("Tugboat episode", win_size[0], win_size[1], autoflush=False)
        ship_rend = render(win, self.environment, self.initial_state, self.px_scale)
        time.sleep(delay)
        for action in self.actions:
            if math.isclose(action.r, 0, rel_tol=1e-5):
                ship_rend.move(math.cos(math.radians(state.r))*action.x, math.sin(math.radians(state.r))*action.x)
            else:
                #IMPLEMENT REDRAWING OF SHIP USING graphics.undraw() and then graphics.draw()
                pass
            graphics.update(1/delay)

def test():
    win = graphics.GraphWin("Tugboats", 1000, 800)
    dom = Domain([[0,0],[0,1],[1,1]])
    dom1 = Domain([[1,1],[2,1],[2,2],[1,2]])
    ship_length = 1
    env = Environment([dom, dom1], ship_length)
    state = State(2,2,20)
    if(state.is_valid(env)):
        px_scale = 80
        render(win, env, state, px_scale)
