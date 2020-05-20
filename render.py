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

    message = graphics.Text(graphics.Point(win.getWidth()/2, 20), 'Click anywhere to quit.')
    message.draw(win)
    win.getMouse()
    win.close()

    return ship_rend


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

        message = Text(Point(win.getWidth()/2, 20), 'Click anywhere to quit.')
        message.draw(win)
        win.getMouse()
        win.close()

        ship_rend = render(win, self.environment, self.initial_state, self.px_scale)
        time.sleep(delay)
        for action in self.actions:
            if math.isclose(action.r, 0, rel_tol=1e-5):
                ship_rend.move(math.cos(math.radians(state.r))*action.x, math.sin(math.radians(state.r))*action.x)
            else:
                #TODO IMPLEMENT REDRAWING OF SHIP USING graphics.undraw() and then graphics.draw()
                pass
            graphics.update(1/delay)

def test():
    from learn import Domain, Environment

    WIN_WIDTH = 500
    WIN_HEIGHT = 500
    win = graphics.GraphWin("Tugboats", WIN_WIDTH, WIN_HEIGHT)
    SHIP_LENGTH = 10
    GRID_HEIGHT = 100       #Define square grid
    GRID_WIDTH = 100
    POLYGONS = [[[0,40],[50,40],[50,50],[0,50]],
                [[40,50],[50,50],[50,80],[40,80]]]
    ship_length = 10
    env = Environment(POLYGONS, SHIP_LENGTH, [GRID_WIDTH, GRID_HEIGHT])
    state = [10,10,0]
    px_scale = min(int(WIN_WIDTH/GRID_WIDTH), int(WIN_HEIGHT/GRID_HEIGHT))
    render(win, env, state, px_scale)
