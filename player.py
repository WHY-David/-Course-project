from consts import Consts
import numpy as np

# from DDPG_keras import agent
from stable_baselines import A2C, DDPG

USING = 'A2C'
NUMCONC = 15

class Player():
    def __init__(self, id, arg=None):
        self.id = id
        if USING=='DDPG':
            self.model = DDPG.load('DDPG_baselines')
        elif USING=='A2C':
            self.model = A2C.load("A2C_baselines")
        # agent.load_weights('OsmoEnv.hdf5')

    def strategy(self, allcells):
        # x, y = agent.forward(self.inputdata(allcells))
        x, y = self.model.predict(self.inputdata(allcells))[0]
        z = x + 1j*y
        return np.pi/2-np.angle(z) if abs(z) >= 1 else None

    def inputdata(self, cells):
        mycell = cells.pop(0)
        opcell  =  cells.pop(0)
        if self.id:
            mycell, opcell = opcell, mycell

        # relative displacement and velocity and radius
        def info(cell):
            x = cell.pos[0]-mycell.pos[0]
            y = cell.pos[1]-mycell.pos[1]
            dx = min(x, x+Consts['WORLD_X'], x-Consts['WORLD_X'], key=abs)
            dy = min(y, y+Consts['WORLD_Y'], y-Consts['WORLD_Y'], key=abs)
            # return [np.array([dx, dy])/mycell.radius, np.array([cell.veloc[0]-mycell.veloc[0], cell.veloc[1]-mycell.veloc[1]])/mycell.radius, cell.radius/mycell.radius]
            return [dx/mycell.radius, dy/mycell.radius, (cell.veloc[0]-mycell.veloc[0])/mycell.radius, (cell.veloc[1]-mycell.veloc[1])/mycell.radius, cell.radius/mycell.radius]

        # cells of concern = nearest cells and opponent
        cells.sort(key=mycell.distance_from)
        if len(cells) < NUMCONC-1:
            data = [info(c) for c in cells]
            while len(data) < NUMCONC-1:
                theta = 2*np.pi*np.random.random()
                data.append([Consts['WORLD_X']/15*np.cos(theta),
                             Consts['WORLD_X']/15*np.sin(theta), 0., 0., 0.])
        else:
            data = [info(c) for c in cells[:NUMCONC-1]]
        data.append(info(opcell))
        return np.array(data)