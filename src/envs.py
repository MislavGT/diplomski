from const import *
from util import *

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
from .fem import Graph, Segment, Topology
from copy import deepcopy


class SegmentEnv(gym.Env):

    def __init__(self, elemsno):

        self.elemsno = elemsno
        self.observation_space = spaces.Box(
            np.array([-np.inf, -np.inf, -np.inf, 0]), 
            np.array([np.inf, np.inf, np.inf, 1]), 
            shape=(4,), dtype=float
        )
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):

        return  np.hstack((np.array([self.sol.cell_jump(self.location)], dtype=float), 
                           np.array([self.sol.neighbor_jump(self.location)], dtype=float), 
                           np.array([self.sol.average_jump()], dtype=float), 
                           np.array([self.sol.K/self.elemsno], dtype=float)
                ))

    def _get_info(self):

        return {
                    "solution": self.sol.u,
                    "elements": self.sol.K
                }

    def reset(self, seed=None, options=None):
        
        self.sol = Segment(1, 1, 0, 2*np.pi, self.elemsno)
        self.sol.run_advec()
        for i in range(self.elemsno//2):
            self.sol.refine(np.random.randint(self.sol.K))
        self.location = np.random.randint(self.sol.K)
        self.actsno = 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        p0 = self.sol.K/self.elemsno
        if action < 1:
            if 1 == (self.sol.K+1)/self.elemsno:
                self.location = np.random.randint(self.sol.K)
                return self._get_obs(), EXCEED, False, self._get_info()
            self.sol.refine(self.location)
            self.actsno += 1
            signed = 1
        elif action < 2:
            if self.sol.coarsen():
                self.location = np.random.randint(self.sol.K)
                return self._get_obs(), 0, False, self._get_info()
            signed = -1
        else:
            return self._get_obs(), 0, False, self._get_info()
        p1 = self.sol.K/self.elemsno
        delta_sol = self.sol.integrate()
        self.location = np.random.randint(self.sol.K)
        reward = signed * (np.log(delta_sol + EPSMACH) - np.log(EPSMACH)) - \
                 GAMMA * (barrier(p1) - barrier(p0))
        terminated = self.actsno == ACTIONS
        return self._get_obs(), reward, terminated, self._get_info()
    

class GraphEnv(gym.Env):

    def __init__(self, T):

        self.begin = deepcopy(T)
        self.location = T.wrap_random()
        self.dnc = 0
        self.cnt = 0
        return  np.hstack((np.array([self.T.cell_jump(self.location)], dtype=float), 
                           np.array([self.T.neighbor_jump(self.location)], dtype=float), 
                           np.array([self.T.average_jump()], dtype=float), 
                           np.array([self.ecnt/self.elemsno], dtype=float)
                ))
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):

        return np.hstack((
            np.array([self.T.jump(self.location)], dtype=float),
            np.array([self.T.neighbor_jump(self.location)], dtype=float),
            np.array([self.T.average_jump()], dtype=float),
            np.array([self.T.ecnt/self.elemsno], dtype=float)
        ))

    def _get_info(self):

        return{
            "elements": self.T.ecnt
        }

    def reset(self, seed=None, options=None):
        
        self.T = deepcopy(self.begin)
        self.dnc = 0
        self.cnt = 10
        for _ in range(10):
            self.T.refine(self.T.wrap_random())
        self.location = self.T.wrap_random()
        self.T.calculate()
        observation = self._get_obs()

        return observation
    
    def _refine(self, n):
        self.dnc = 0
        integral = self.T.integrate_difference(n, True)
        return self._calc_rew(integral)
    
    def _coarsen(self, n):
        integral = self.T.integrate_difference(n, False)
        if not integral:
            return self._nothing()
        self.dnc = 0
        return -self._calc_rew(integral)
        
    def _nothing(self):
        self.dnc += 1
        return 0
    
    def _calc_rew(self, integral):
        return np.log(integral+EPSMACH) - np.log(EPSMACH)
    
    def _comp_cost(self, prev):
        return barrier(self.T.ecnt/(ELEMSNO+1)) - barrier(prev/(ELEMSNO+1))
    
    def step(self, action):
        prev = self.T.ecnt
        if action < 1:
            if self.T.ecnt == ELEMSNO:
                self.location = self.T.wrap_random()
                self.dnc += 1
                self.cnt += 1
                return self._get_obs(), EXCEED, False, self._get_info()
            integral = self._refine(self.location)
        elif action < 2:
            integral = self._coarsen(self.location)
        else:
            integral = self._nothing()
        self.cnt += 1
        terminated = self.cnt == ACTIONS
        if not integral:
            reward = 0
        else:
            reward = integral - GAMMA * self._comp_cost(prev)

        self.location = self.T.wrap_random()
        return self._get_obs(), reward, terminated, self._get_info()

class HeatEnv(gym.Env):

    def __init__(self, design: Topology):
        self.counter = 0
        self.begin = deepcopy(design)
        self.observation_space = spaces.Box(
            low = -np.inf * np.ones(design.features.shape[0]*(design.size+1)+14*design.size,),
            high = np.inf * np.ones(design.features.shape[0]*(design.size+1)+14*design.size,),
            dtype = float
        )
        self.action_space = spaces.Discrete(design.size-1)

    def _get_obs(self):
        return np.hstack((self.design.features.flatten(), 
                          self.design.senders, 
                          self.design.receivers))
    
    def _get_info(self):
        return{
            "invalid": None
        }

    def reset(self, seed=None, options=None):
        self.counter = 0
        self.design = deepcopy(self.begin)
        self.design.calculate()
        return self._get_obs(), self._get_info()
    
    def _reward(self):
        return (-self.design.u @ self.design.F)/(len(self.design.F)**2)

    def step(self, action):

        self.design.increase(action)
        self.design.calculate()
        self.counter += 1

        if self.counter == 100:
            return self._get_obs(), self._reward(), True, False, self._get_info()
        
        return self._get_obs(), 0, False, False, self._get_info()