import numpy as np
from random import uniform

class Fourrooms:
    def __init__(self, initstate_seed, punishEachStep, modified):
        self.punishEachStep = punishEachStep
        self.modified = modified

        self.layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""

        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in self.layout.splitlines()])

        self.action_space = 4

        self.observation_space = int(np.sum(self.occupancy == 0))

        # 0 - Up
        # 1 - Down
        # 2 - Left
        # 3 - Right

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]

        self.rng = np.random.RandomState(1234)

        self.initstate_seed = initstate_seed
        self.rng_init_state = np.random.RandomState(self.initstate_seed)

        self.tostate = {}

        self.occ_dict = dict(zip(range(self.observation_space),
                                 np.argwhere(self.occupancy.flatten() == 0).squeeze()))

        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1

        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62
        self.init_states = list(range(self.observation_space))
        self.init_states.remove(self.goal)


    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space):
            nextcell = tuple(cell + np.multiply(self.directions[action], self.inQuad24(self.currentcell)))
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self, test=None):
        if test:
            state = test
        else:
            state = self.rng_init_state.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return state

    def step(self, action):
        reward = -2 * int(self.punishEachStep)
        if self.rng.uniform() < 1/3:
            empty_cells = self.empty_around(self.currentcell)
            nextcell = empty_cells[self.rng.randint(len(empty_cells))]
        else:
            nextcell = tuple(self.currentcell + np.multiply(self.directions[action], self.inQuad24(self.currentcell)))

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]

        if state == self.goal:
            reward = 50

        done = state == self.goal
        return state, reward, float(done), None

    def inQuad24(self, cell):
        if not(self.modified):
            return np.array([1,1])
        if cell[1] > 6:
            if cell[0] < 7:
                return np.array([1, -1])
        else:
            if cell[0] > 6:
                return np.array([1, -1])
        return np.array([1, 1])