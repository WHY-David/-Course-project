# Change at line 104

import numpy as np
from copy import deepcopy
import gym
from gym import spaces

from consts import Consts
from worldfortraining import World
from sample.brownian_motion import Player as Opponent

NUMCONC=15

WINREWARD = 10.
LOSEPENALTY = 10.
RADIUSREWARD = 0.5
TIMEREWARD = 1.06
KILLREWARD = 0.5
EJECTPENALTY = 600.9

class dummyPlayer:
    def __init__(self, id, arg=None):
        self.id=id

class OsmoEnv(World,gym.Env):
    def __init__(self):
        super(OsmoEnv,self).__init__(dummyPlayer(0),Opponent(1))
        self.action_space=spaces.Box(low=-5e0,high=5e0,shape=(2,))
        self.observation_space=spaces.Box(low=-1e4,high=1e4,shape=(NUMCONC,5))

    def reset(self):
        self.new_game()
        return self.observe()

    def step(self,action:np.array): # -> observation(np.array), reward(float), done(bool), info(dict or None)
        # assert self.action_space.contains(action), '%r (%s) invalid action'%(action,type(action))
        
        # below are almost the same as World.update()
        self.frame_count += 1
        if self.frame_count == Consts["MAX_FRAME"]: # Time's up
            self.check_point(self.cells[0].radius <= self.cells[1].radius, self.cells[0].radius >= self.cells[1].radius, "MAX_FRAME")
            return self.observe(), -LOSEPENALTY, True, self.result
        for cell in self.cells:
            if not cell.dead:
                cell.move(Consts['FRAME_DELTA'])
        # Detect collisions
        collisions = []
        for i in range(len(self.cells)):
            if self.cells[i].dead:
                continue
            for j in range(i + 1, len(self.cells)):
                if not self.cells[j].dead and self.cells[i].collide(self.cells[j]):
                    if self.cells[i].collide_group == None == self.cells[j].collide_group:
                        self.cells[i].collide_group = self.cells[j].collide_group = len(collisions)
                        collisions.append([i, j])
                    elif self.cells[i].collide_group != None == self.cells[j].collide_group:
                        collisions[self.cells[i].collide_group].append(j)
                        self.cells[j].collide_group = self.cells[i].collide_group
                    elif self.cells[i].collide_group == None != self.cells[j].collide_group:
                        collisions[self.cells[j].collide_group].append(i)
                        self.cells[i].collide_group = self.cells[j].collide_group
                    elif self.cells[i].collide_group != self.cells[j].collide_group:
                        collisions[self.cells[i].collide_group] += collisions[self.cells[j].collide_group]
                        for ele in collisions[self.cells[j].collide_group]:
                            self.cells[ele].collide_group = self.cells[i].collide_group
                        collisions[self.cells[j].collide_group] = []
        # Run absorbs
        for collision in collisions:
            if collision != []:
                self.absorb(collision)
        # If we just killed the player, Game over
        if self.check_point(self.cells[0].dead, self.cells[1].dead, "PLAYER_DEAD"):
            return self.observe(), self.reward()+(WINREWARD if self.cells[1].dead else -LOSEPENALTY), True, self.result
        # Eject!
        allcells = [cell for cell in self.cells if not cell.dead]
        # assert allcells[0].id == 0, 'allcells[0] is not me!'
        self.cells_count = len(allcells)
        theta0 = theta1 = None
        flag0 = flag1 = False

        if self.timer[0] > 0:
            try:
                z=action[0]+1j*action[1]
                if abs(z)>=1:
                    theta0=np.angle(z)
                    self.ejectnum+=1
            except Exception as e:
                flag0 = e
        if self.timer[1] > 0:
            try:
                theta1 = self.player1.strategy(deepcopy(allcells))
            except Exception as e:
                flag1 = e

        if isinstance(theta0, (int, float, type(None))):
            if self.timer[0] >= 0:
                self.eject(self.cells[0], theta0)
        else:
            flag0 = True
        if isinstance(theta1, (int, float, type(None))):
            if self.timer[1] >= 0:
                self.eject(self.cells[1], theta1)
        else:
            flag1 = True
        self.check_point(flag0, flag1, "RUNTIME_ERROR")

        return self.observe(), self.reward(), self.result!={}, self.result

    # for visualization
    def render(self):
        pass
    
    # close GUI window
    def close(self):
        pass

    def observe(self): # -> np.array (NUMCONC,5)
        allcells=[cell for cell in self.cells if not cell.dead]
        mycell=allcells.pop()
        opcell=allcells.pop()
        if self.player0.id:
            mycell,opcell = opcell,mycell

        # relative displacement and velocity and radius
        def info(cell):
            x = cell.pos[0]-mycell.pos[0]
            y = cell.pos[1]-mycell.pos[1]
            dx = min(x, x+Consts['WORLD_X'], x-Consts['WORLD_X'], key=abs)
            dy = min(y, y+Consts['WORLD_Y'], y-Consts['WORLD_Y'], key=abs)
            temp=[dx, dy, cell.veloc[0]-mycell.veloc[0], cell.veloc[1]-mycell.veloc[1], cell.radius]
            return [x/mycell.radius for x in temp]

        # cells of concern = nearest cells and opponent
        allcells.sort(key=mycell.distance_from)
        if len(allcells) < NUMCONC-1:
            data = [info(c) for c in allcells]
            # fill up
            while len(data) < NUMCONC-1:
                theta = 2*np.pi*np.random.random()
                data.append([Consts['WORLD_X']/15*np.cos(theta), Consts['WORLD_X']/15*np.sin(theta), 0., 0., 0.])
        else:
            data = [info(c) for c in allcells[:NUMCONC-1]]
        data.append(info(opcell))
        return np.array(data)

    def reward(self):
        return RADIUSREWARD*self.cells[0].radius\
            + TIMEREWARD*self.frame_count\
            + KILLREWARD*self.player0absorbed\
            - EJECTPENALTY*self.ejectnum