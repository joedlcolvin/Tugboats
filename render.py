import graphics
import math
import time

def render(win, initial_state, px_scale, environment, enviro_rend=False):
    if enviro_rend:
        for polygon in environment.polygons:
            for i in range(len(polygon.e)):
                graphics.Line(  graphics.Point(polygon.e[i][0][0]*px_scale, polygon.e[i][0][1]*px_scale), 
                                graphics.Point(polygon.e[i][1][0]*px_scale, polygon.e[i][1][1]*px_scale)).draw(win)

    ship = environment.ship_ends(initial_state)
    ship_rend = graphics.Line(  graphics.Point(ship[0][0]*px_scale, ship[0][1]*px_scale), 
                                graphics.Point(ship[1][0]*px_scale, ship[1][1]*px_scale)).draw(win)

    return ship_rend

def render_reward(win, reward):
    return graphics.Text(graphics.Point(7*win.getWidth()/8, win.getHeight()-20), 'Reward: ' + str(reward)).draw(win)

class Episode:
    def __init__(self, states, rewards, environment, px_scale, win_width, win_height):
        self.environment = environment
        self.states = states
        self.rewards = [rewards[0]]
        for i in range(len(rewards)-1):
            self.rewards.append(rewards[i+1]+self.rewards[i])
        self.px_scale = px_scale
        self.win_width = win_width
        self.win_height = win_height
        self.ship_rend = None
        self.reward_rend = None

    def animate(self, delay):
        win = graphics.GraphWin("Tugboat episode", self.win_width, self.win_height, autoflush=False)
        message = graphics.Text(graphics.Point(win.getWidth()/2, 20), 'Click anywhere to play.')
        message.draw(win)
        self.ship_rend = render(win, self.states[0], self.px_scale, self.environment, True)
        self.reward_rend = render_reward(win, 0)
        while(True):
            win.getMouse()
            self.ship_rend.undraw()
            self.reward_rend.undraw()
            self._play_episode(win, delay)

    def _play_episode(self, win, delay):
        # We render ship twice on first state, but this doesn't matter
        for i, state in enumerate(self.states):
            self.ship_rend.undraw()
            self.reward_rend.undraw()
            self.ship_rend = render(win, state, self.px_scale, self.environment)
            self.reward_rend = render_reward(win, self.rewards[i])
            graphics.update(delay)

def test_static():
    from learn import Polygon, Environment

    WIN_WIDTH = 500
    WIN_HEIGHT = 500
    win = graphics.GraphWin("Tugboats", WIN_WIDTH, WIN_HEIGHT)
    SHIP_LENGTH = 10
    GRID_HEIGHT = 100       #Define square grid
    GRID_WIDTH = 100
    POLYGONS = [[[0,40],[50,40],[50,50],[0,50]], [[40,50],[50,50],[50,80],[40,80]]]
    ship_length = 10
    env = Environment(POLYGONS, SHIP_LENGTH, [GRID_WIDTH, GRID_HEIGHT], 50, 1, [1,1,1,1,1,1], 10, [0,0,0], [0,0,0], 6)
    state = [10,10,0]
    px_scale = min(int(WIN_WIDTH/GRID_WIDTH), int(WIN_HEIGHT/GRID_HEIGHT))
    render(win, state, px_scale, env)

def test_episode():
    from learn import Environment
    
    WIN_WIDTH = 500
    WIN_HEIGHT = 500
    SHIP_LENGTH = 10
    GRID_HEIGHT = 100       #Define square grid
    GRID_WIDTH = 100
    POLYGONS = [[[0,40],[50,40],[50,50],[0,50]], [[40,50],[50,50],[50,80],[40,80]]]
    ship_length = 10
    env = Environment(POLYGONS, SHIP_LENGTH, [GRID_WIDTH, GRID_HEIGHT], 50, 1, [1,1,1,1,1,1], 10, [0,0,0], [0,0,0], 6)
    states = [[10,10,i] for i in range(360)]
    rewards = [-1 for i in range(360)]
    px_scale = min(int(WIN_WIDTH/GRID_WIDTH), int(WIN_HEIGHT/GRID_HEIGHT))
    episode = Episode(states, rewards, env, px_scale, WIN_WIDTH, WIN_HEIGHT)
    episode.animate(60)
