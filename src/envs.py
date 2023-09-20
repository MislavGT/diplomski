from const import *
from util import *

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
from .fem import Graph, Segment, Topology
from copy import deepcopy
from scipy.sparse.linalg import ArpackNoConvergence


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
        self.cnt = 0
        self.observation_space = spaces.Box(
            low = -np.inf * np.ones((4,)),
            high = np.inf * np.ones((4,)),
            dtype = float
        )
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):

        return np.hstack((
            np.array([self.T.jump(self.location)], dtype=float),
            np.array([self.T.neighbor_jump(self.location)], dtype=float),
            np.array([self.T.average_jump()], dtype=float),
            np.array([self.T.ecnt/ELEMSNO], dtype=float)
        ))

    def _get_info(self):

        return{
            "elements": self.T.ecnt
        }

    def reset(self, seed=None, options=None):
        self.T = deepcopy(self.begin)
        self.cnt = 0
        for i in range(4):
            self.T.refine(i)
        for i in range(8):
            self.T.refine(i)
        self.location = np.random.randint(0, self.T.ecnt)
        self.T.calculate()

        return self._get_obs(), self._get_info()
    
    def _refine(self, n):
        integral = self.T.integrate_difference(n, True)
        return self._calc_rew(integral)
    
    def _coarsen(self, n):
        integral = self.T.integrate_difference(n, False)
        if not integral:
            return self._nothing()
        return -self._calc_rew(integral)
        
    def _nothing(self):
        return 0
    
    def _calc_rew(self, integral):
        return np.log(integral+EPSMACH) - np.log(EPSMACH)
    
    def _comp_cost(self, prev):
        return barrier(self.T.ecnt/(ELEMSNO)) - barrier(prev/(ELEMSNO))
    
    def step(self, action):
        self.cnt += 1
        prev = self.T.ecnt
        if action < 1:
            if self.T.ecnt+1 == ELEMSNO:
                self.location = np.random.randint(0, self.T.ecnt-1)
                self.cnt += 1
                return self._get_obs(), EXCEED, False, False, self._get_info()
            integral = self._refine(self.location)
        elif action < 2:
            integral = self._coarsen(self.location)
        else:
            integral = self._nothing()
        terminated = self.cnt == ACTIONS
        if not integral:
            reward = 0
        else:
            reward = integral - GAMMA * self._comp_cost(prev)

        self.location = np.random.randint(0, self.T.ecnt-1)
        return self._get_obs(), reward, terminated, False, self._get_info()

class HeatEnv(gym.Env):

    def __init__(self, A, fun, cond):
        self.A = A
        self.fun = fun
        self.cond = cond
        rnd = np.sort(np.random.uniform(1, 10, 4))
        coord = np.vstack(i.flatten() for i in np.meshgrid(np.linspace(rnd[0], rnd[2], 4), 
                                                           np.linspace(rnd[1], rnd[3], 4))).T
        design = Topology(self.A, coord, self.fun, self.cond)
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
        rnd = np.sort(np.random.uniform(1, 10, 4))
        coord = np.vstack(i.flatten() for i in np.meshgrid(np.linspace(rnd[0], rnd[2], 4), 
                                                           np.linspace(rnd[1], rnd[3], 4))).T
        self.design = Topology(self.A, coord, self.fun, self.cond)
        self.counter = 0
        self.design.calculate()
        return self._get_obs(), self._get_info()
    
    def reward(self):
        return self.design.n - self.design.o

    def step(self, action):
        if self.design.C[action] == 0.11:
            return self._get_obs(), EXCEED, False, False, self._get_info()
        self.design.increase(action)
        self.design.calculate()
        self.counter += 1

        if self.counter == DEPTH:
            return self._get_obs(), self.reward(), True, False, self._get_info()
        
        return self._get_obs(), self.reward(), False, False, self._get_info()
    

class EigEnv(gym.Env):

    def __init__(self, graph, coord):
        self.ep = -1
        self.graph = graph
        self.coord = coord
        self.observation_space = spaces.Box(
            low = -np.inf * np.ones((4,)),
            high = np.inf * np.ones((4,)),
            dtype = float
        )
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):

        return np.hstack((
            np.array([self.T.jump(self.location)], dtype=float),
            np.array([self.T.neighbor_jump(self.location)], dtype=float),
            np.array([self.T.average_jump()], dtype=float),
            np.array([self.T.ecnt/ELEMSNO], dtype=float)
        ))

    def _get_info(self):

        return{
            "elements": self.T.ecnt
        }

    def reset(self, seed=None, options=None):
        self.ep += 1
        self.T = Graph(self.graph, self.coord, eig=self.ep % 10) # eigenvalue number from linalg.eigs
        self.cnt = 0
        for i in range(self.T.ecnt):
            self.T.refine(i)
        self.location = np.random.randint(0, self.T.ecnt)
        self.T.calculate_eigen()

        return self._get_obs(), self._get_info()
    
    def _refine(self, n):
        integral = self.T.integrate_difference_eig(n, True)
        return self._calc_rew(integral)
    
    def _coarsen(self, n):
        integral = self.T.integrate_difference_eig(n, False)
        if not integral:
            return self._nothing()
        return -self._calc_rew(integral)
        
    def _nothing(self):
        return 0
    
    def _calc_rew(self, integral):
        return np.log(integral+EPSMACH) - np.log(EPSMACH)
    
    def _comp_cost(self, prev):
        return barrier(self.T.ecnt/(ELEMSNO)) - barrier(prev/(ELEMSNO))
    
    def step(self, action):
        self.cnt += 1
        prev = self.T.ecnt
        if action < 1:
            if self.T.ecnt+1 == ELEMSNO:
                self.location = np.random.randint(0, self.T.ecnt-1)
                self.cnt += 1
                return self._get_obs(), EXCEED, False, False, self._get_info()
            try:
                integral = self._refine(self.location)
            except ArpackNoConvergence:
                print('fail')
                return np.array([0, 0, 0, 0]), 0, True, False, self._get_info()
        elif action < 2:
            try:
                integral = self._coarsen(self.location)
            except ArpackNoConvergence:
                print('fail')
                return np.array([0, 0, 0, 0]), 0, True, False, self._get_info()
        else:
            integral = self._nothing()
        terminated = self.cnt == ACTIONS
        if not integral:
            reward = 0
        else:
            reward = integral - GAMMA * self._comp_cost(prev)

        self.location = np.random.randint(0, self.T.ecnt-1)
        return self._get_obs(), reward, terminated, False, self._get_info()