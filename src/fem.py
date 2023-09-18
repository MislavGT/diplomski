from __future__ import annotations

from const import *
from util import *

import numpy as np
import scipy.sparse as ss
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy as sp
from math import gamma
from numpy.linalg import eig, inv
from scipy.sparse import lil_matrix, eye
from copy import deepcopy
from functools import partial
from typing import Callable

class Node():
    
    def __init__(self,
                 n: int = -1,
                 sib: Node = None,
                 leaf: int = -1,
                 parent: Node = None):
        
        self.n = n
        self.sib = sib
        self.leaf = leaf
        self.parent = parent
    
class Segment:

    def __init__(self, N, K, xmin, xmax, elemsno):

        self.elemsno = elemsno
        self.N = N
        self.K = K
        self.levels = [Node(i) for i in range(K)]
        self.vrh = []
        self.xmin = xmin
        self.xmax = xmax
        self.Nfp = 1
        self.Np = N + 1
        self.Nfaces = 2
        self.Nv = K + 1
        self.VX, self.EToV = self.MeshGen1D()
        self.r = JacobiGL(0, 0, self.N)
        self.V = self.Vandermonde1D()
        self.Dvr = self.GradVandermonde1D()
        self.Dr = self.GradVandermonde1D() @ inv(self.V)
        self.LIFT = self.Lift1D()
        va = self.EToV[:self.K, 0].T
        vb = self.EToV[:self.K, 1].T
        self.x = np.outer(np.ones((self.N+1,)), self.VX[va]) + \
                 0.5 * np.outer(self.r+1, self.VX[vb]-self.VX[va])
        self.rx, self.J = self.GeometricFactors1D() 
        self.Fx, self.Fmask, self.fmask1, self.fmask2 = self.Mask()
        self.nx = self.Normals1D()
        self.Fscale = 1./self.J[self.Fmask[0, :], :]
        self.EToE, self.EToF = self.Connect1D()
        self.vmapM, self.vmapP, self.vmapB, self.mapB = self.BuildMaps1D()
        self.mapI = 0
        self.mapO = self.K * self.Nfaces - 1
        self.vmapI = 0
        self.vmapO = self.K * self.Np - 1
        self.levels = list([Node(i) for i in range(self.K)])
        self.run_advec()

    def Vandermonde1D(self):

        V1D = np.zeros((len(self.r), self.N + 1))
        
        for i in range(self.N + 1):
            V1D[:, i] = JacobiP(self.r, 0, 0, i)

        return V1D

    def GradVandermonde1D(self):

        DVr = np.zeros((len(self.r), self.N + 1))

        for i in range(self.N+1):
            DVr[:, i] = GradJacobiP(self.r, 0, 0, i)

        return DVr

    def Lift1D(self):

        Emat = np.zeros((self.Np, self.Nfaces*self.Nfp))
        Emat[0, 0] = 1.
        Emat[self.Np - 1, 1] = 1.
        
        return self.V @ (self.V.T @ Emat)

    def GeometricFactors1D(self):

        xr = self.Dr @ self.x
        
        return 1./xr, xr    

    def Normals1D(self):

        nx = np.zeros((self.Nfp * self.Nfaces, self.K))
        nx[0, :] = -1.
        nx[1, :] = 1.
        
        return nx

    def Connect1D(self):

        TotalFaces = self.Nfaces * self.K
        vn = np.array([0, 1])
        SpFToV = lil_matrix((TotalFaces, self.Nv), dtype = float)
        sk = 0
        
        for k in range(self.K):
            for face in range(self.Nfaces):
                SpFToV[sk, self.EToV[k, vn[face]]] = 1.
                sk+=1

        SpFToF = SpFToV @ SpFToV.T - eye(TotalFaces, \
                                         dtype = float, format = 'csr')
        temp = np.argwhere(SpFToF == 1)
        faces1, faces2 = temp[:, 0], temp[:, 1]
        element1 = faces1 // self.Nfaces
        element2 = faces2 // self.Nfaces
        face1 = faces1 % self.Nfaces
        face2 = faces2 % self.Nfaces
        temp = np.arange(self.K, dtype=int)
        EToE = np.vstack((temp, temp)).T
        EToF = np.vstack((np.zeros((self.K,), dtype=int), \
                          np.ones((self.K,), dtype=int))).T
        EToE[element1, face1] = element2[:]
        EToF[element1, face1] = face2[:]

        return EToE, EToF  

    def Mask(self):

        fmask1 = np.nonzero(np.abs(self.r+1) < NODETOL)
        fmask2 = (np.abs(self.r-1) < NODETOL).nonzero()
        Fmask = np.vstack((fmask1, fmask2)).T
        Fx = self.x[Fmask.flatten(), :]

        return Fx, Fmask, fmask1, fmask2

    def BuildMaps1D(self):

        nodeids = np.arange(self.K*self.Np).reshape(self.K, self.Np).T
        vmapM = np.zeros((self.Nfp, self.Nfaces, self.K), dtype=int)
        vmapP = np.zeros((self.Nfp, self.Nfaces, self.K), dtype=int)
        temp = self.x.reshape(self.K*self.Np, order='F')

        for i in range(self.K):
            for j in range(self.Nfaces):
                vmapM[:, j, i] = nodeids[self.Fmask[:, j], i]
        for i in range(self.K):
            for j in range(self.Nfaces):
                k2 = self.EToE[i, j]
                f2 = self.EToF[i, j]
                vidM = vmapM[:, j, i]
                vidP = vmapM[:, f2, k2]
                x1 = temp[vidM]
                x2 = temp[vidP]
                D = np.inner(x1 - x2, x1 - x2)
                
                if D < NODETOL:
                    vmapP[:, j, i] = vidP

        vmapP = vmapP.reshape(self.Nfp*self.Nfaces*self.K, order='F')
        vmapM = vmapM.reshape(self.Nfp*self.Nfaces*self.K, order='F')
        mapB = np.where(vmapP == vmapM)
        vmapB = vmapM[mapB]

        return vmapM, vmapP, vmapB, mapB
        
    def MeshGen1D(self):
        
        VX = np.empty(self.elemsno + 1)
        VX[:self.Nv] = np.linspace(self.xmin, self.xmax, self.Nv)
        EToV = np.empty((self.elemsno, 2), dtype=int)
        EToV[:self.K, :] = np.vstack((np.arange(self.K), np.arange(self.K) + 1)).T
        self.vrh = [[0]]
        for i in range(self.K-1):
            self.vrh.append([i, i+1])
        self.vrh.append([self.K-1])

        return VX, EToV

    def AdvecRHS1D(self):

        alpha = 1
        memshape = self.u.shape
        memshape3 = self.nx.shape
        self.u = self.u.ravel('F')
        du = (self.u[self.vmapM] - self.u[self.vmapP])
        du = du.reshape(self.nx.shape[1], self.nx.shape[0]).T
        du *= (self.a * self.nx - (1 - alpha) * np.abs(self.a * self.nx))/2
        uin = -np.sin(self.a * self.timelocal)
        memshape2 = du.shape
        du = du.ravel('F')
        self.nx = self.nx.ravel('F')
        du[self.mapI] = (self.u[self.vmapI] - uin) * (self.a * self.nx[self.mapI] - \
                        (1 - alpha) * np.abs(self.a * self.nx[self.mapI]))/2
        du[self.mapO] = 0
        du = du.reshape(memshape2[1], memshape2[0]).T
        self.u = self.u.reshape(memshape[1], memshape[0]).T
        self.nx = self.nx.reshape(memshape3[1], memshape3[0]).T
        
        self.rhsu = -self.a * self.rx * (self.Dr @ self.u) + \
                    self.LIFT @ (self.Fscale * du)
        
    def Advec1D(self):

        self.time = 0
        self.resu = np.zeros((self.Np, self.K))
        minx = np.min(np.abs(self.x[0, :] - self.x[1, :]))
        CFL = 0.75
        dt = CFL / (2 * np.pi) * minx / 2
        Nsteps = int(np.ceil(self.FinalTime / dt))
        dt = self.FinalTime / Nsteps
        self.a = 2 * np.pi

        for i in range(Nsteps):
            for j in range(5):
                self.timelocal = self.time + rk4c[j]*dt
                self.AdvecRHS1D()
                self.resu = rk4a[j] * self.resu + dt * self.rhsu
                self.u += rk4b[j] * self.resu
            self.time += dt

    def run_advec(self):
        
        self.LIFT = self.Lift1D()
        va = self.EToV[:self.K, 0].T
        vb = self.EToV[:self.K, 1].T
        self.x = np.outer(np.ones((self.N+1,)), self.VX[va]) + \
                 0.5 * np.outer(self.r+1, self.VX[vb]-self.VX[va])
        self.rx, self.J = self.GeometricFactors1D() 
        self.Fx, self.Fmask, self.fmask1, self.fmask2 = self.Mask()
        self.nx = self.Normals1D()
        self.Fscale = 1./self.J[self.Fmask[0, :], :]
        self.EToE, self.EToF = self.Connect1D()
        self.vmapM, self.vmapP, self.vmapB, self.mapB = self.BuildMaps1D()
        self.mapO = self.K * self.Nfaces - 1
        self.vmapO = self.K * self.Np - 1
        self.u = np.sin(self.x)
        self.FinalTime = 10
        self.Advec1D()

        return self.u

    def refine(self, i):

        self.oldu = np.hstack((self.u, self.u[:, np.min(self.EToV[i])].reshape(self.u.shape)))
        self.vrh[self.EToV[i][1]][0] = self.K
        self.vrh.append([i, self.K])
        self.VX[self.K+1] = np.sum(self.VX[self.EToV[i]])/2
        self.EToV[self.K] = np.array([self.K+1, self.EToV[i][1]])
        self.EToV[i] = np.array([self.EToV[i][0], self.K+1])
        self.K += 1
        self.Nv += 1
        self.levels.append(Node(self.K-1))
        self.levels[i].leaf = 0
        self.levels[i] = Node(i, None, 1, self.levels[i])
        self.levels[self.K-1] = Node(self.K-1, self.levels[i], 1, self.levels[i].parent)
        self.levels[i].sib = self.levels[self.K-1]

        self.run_advec()
        self.oldx = deepcopy(self.x)
        self.newu = deepcopy(self.u)

    def coarsen(self, i):
        self.oldx = deepcopy(self.x)
        self.oldu = deepcopy(self.u)
        if i < 0 and i > self.K-1:
            return True
        try:
            if not (self.levels[i].leaf == 1 and self.levels[i].sib.leaf == 1):
                return True
        except AttributeError:
            return True

        n = min(i, self.levels[i].sib.n)
        m = max(i, self.levels[i].sib.n)
        t1 = self.EToV[n][0]
        t2 = self.EToV[m][1]
        t = self.EToV[n][1]
        self.EToV[n] = np.array([self.EToV[n][0], self.EToV[self.levels[n].sib.n][1]])
        self.EToV[m] = self.EToV[self.K-1]
        self.levels[n] = self.levels[n].parent
        self.levels[n].leaf = 1
        self.levels[n].n = n
        try:
            self.levels[m] = self.levels.pop()
        except IndexError:
            self.levels.pop()
        try:
            self.vrh[t] = self.vrh.pop()
        except IndexError:
            self.vrh.pop()
        for x in self.vrh[t]:
            for y in range(2):
                if self.EToV[x][y] == self.Nv-1:
                    self.EToV[x][y] = t
        self.VX[t] = self.VX[self.Nv-1]
        self.K -= 1
        self.Nv -= 1

        self.run_advec()
        self.newu = np.hstack((self.u, self.u[:, t]))
        self.newu[t] = (self.u[:, t1] + self.u[:, t2])/2
        return False

    def integrate(self):

        d = np.abs(np.diff(self.oldx, axis=0))
        s = self.newu - self.oldu
        oi = 0
        for i in range(d.shape[1]):
            for j in range(d.shape[0]):
                if s[j, i] * s[j+1, i] < 0:
                    oi += ((s[j, i]**2 + s[j+1, i]**2)*d[j, i]/2/(np.abs(s[j, i])+np.abs(s[j+1, i])))
                else:
                    oi += (s[j, i]+s[j, i+1])*d[j, i]/2
        return oi

    def average_jump(self):

        x = 0
        for i in range(self.K):
            x += self.cell_jump(i)
        return x/self.K

    def cell_jump(self, n):

        j = 0
        t1 = self.EToV[n][0]
        flag = False
        for i in self.vrh[t1]:
            if i != t1:
                t2 = i
                flag = True
                break
        if flag:
            if self.x[-1, t2] == self.u[0, t1]:
                j += np.abs(self.u[-1, t2] - self.u[0, t1])
            else:
                j += np.abs(self.u[0, t2] - self.u[-1, t1])
        t1 = self.EToV[n][1]
        flag = False
        for i in self.vrh[t1]:
            if i != t1:
                t2 = i
                flag = True
                break
        if flag:
            if self.x[-1, t2] == self.u[0, t1]:
                j += np.abs(self.u[-1, t2] - self.u[0, t1])
            else:
                j += np.abs(self.u[0, t2] - self.u[-1, t1])
        return j

    def neighbor_jump(self, n):

        r = 0
        vrhovi = self.EToV[n]
        for i in vrhovi:
            bridovi = self.vrh[i]
            for j in bridovi:
                if j != n:
                    r += self.cell_jump(j)
        return r

class Graph:

    def __init__(self,
                 graph: ss.spmatrix,
                 coord: np.ndarray = None,
                 potential: Callable[[np.ndarray], float] = lambda x: 1,
                 fun: Callable[[np.ndarray], float] = lambda x: 1,
                 elimit: int = -1,
                 default: np.ndarray = None
                 ) -> None:
        
        self.fun = fun
        self.vfun = jax.vmap(fun)
        self.potential = potential  
        self.vpot = jax.vmap(potential)
        if type(coord) == type(None):
            coord = np.zeros((graph.shape[0], 3))
        if elimit == -1:
            self.inc = max(graph.shape[0], graph.shape[1])
        else:
            self.inc = elimit - graph.shape[1]
        self.default = default
        self.origv = graph.shape[0]
        self.orige = graph.shape[1]
        self.ecnt = graph.shape[1]
        self.vcnt = graph.shape[0]
        self.R = []
        self.graph = ss.coo_array((graph.data,
                                   (graph.row,
                                    graph.col)))
        self.offset = graph.shape[1] - graph.shape[0]
        self.E = ss.csc_array((graph.data, (graph.row, graph.col)), 
                              shape=(self.inc+graph.shape[0], 
                                     self.inc+graph.shape[1]))
        self.level = [Node(i) for i in range(graph.shape[1])]
        self.coord = np.vstack((coord, np.empty((self.inc, 3))))
        temp = coord[self.E[:self.vcnt, :self.ecnt].indices]
        polo = (temp[::2] + temp[1::2])/2
        temp = temp[::2] - temp[1::2]
        self.S = np.vstack((temp, np.empty((self.inc, 3))))
        self.B = la.norm(temp, axis=1)
        self.W = ss.spdiags(self.B, diags=0, m=(self.E.shape[1], 
                                         self.E.shape[1]), format='csc')
        self.D = ss.spdiags(1/self.B, diags=0, m=(self.E.shape[1], 
                                         self.E.shape[1]), format='csc')
        self.P = np.hstack((self.vpot(coord),
                            np.empty((self.inc,))))
        self.K = ss.spdiags(self.vpot(polo), 
                            diags=0, m=(self.E.shape[1], 
                                         self.E.shape[1]), format='csc')
        vert = ss.csr_array((graph.data, 
                             (graph.row, 
                              graph.col)))
        self.V = []
        mx = 0
        indptr = vert.indptr
        indices = vert.indices
        for i in range(len(indptr)-1):
            self.V.append(list(indices[indptr[i]:indptr[i+1]]))
            if len(self.V[i]) > mx:
                mx = len(self.V[i])
        self.max = 2 * (mx - 1)

    def increase(self):

        self.inc *= 2
        self.E.resize((self.E.shape[0]+self.inc, self.E.shape[1]+self.inc))
        self.coord = np.vstack((self.coord, np.empty((self.inc, 3))))
        self.S = np.vstack((self.S, np.empty((self.inc, 3))))
        self.W.resize((self.E.shape[1], self.E.shape[1]))
        self.D.resize((self.E.shape[1], self.E.shape[1]))
        self.P = np.hstack((self.P, np.empty((self.inc,))))
        self.K.resize((self.E.shape[1], self.E.shape[1]))
        
    def refine(self, n: int=0):

        vrhovi = self.E[:, [n]].indices
        
        if self.R:
            vn, en = self.R.pop()
            flag = False
        else:
            vn, en = self.vcnt, self.ecnt
            if self.E.shape[0] == self.vcnt or self.E.shape[1] == self.ecnt:
                self.increase()
            self.vcnt += 1
            self.ecnt += 1
            flag = True

        self.E[vn, n] = self.E[vrhovi[1], n]
        self.E[vrhovi[1], n] = 0
        self.E[vrhovi[1], en] = 1
        self.E[vn, en] = -1
        self.level[n].leaf = min(0, self.level[n].leaf)
        self.level[n] = Node(n, None, 1, self.level[n])
        self.V[vrhovi[1]].remove(n)
        self.V[vrhovi[1]].append(en)
        if flag:
            self.level.append(Node(en, self.level[n], 1, self.level[n].parent))
            self.V.append([n, en])
        else:
            self.level[en] = Node(en, self.level[n], 1, self.level[n].parent)
            self.V[vn] = [n, en]
        self.level[n].sib = self.level[en]
        self.coord[vn] = (self.coord[vrhovi[0]] + self.coord[vrhovi[1]]) / 2
        self.W[n, n] /= 2
        self.W[en, en] = self.W[n, n]
        self.D[n, n] *= 2
        self.D[en, en] = self.D[n, n]
        temp = self.E[:, [n]].indices
        self.S[n] = self.coord[temp[0]] - self.coord[temp[1]]
        temp = self.E[:, [en]].indices
        self.S[en] = self.coord[temp[0]] - self.coord[temp[1]]
        self.P[vn] = self.K[n, n]
        self.K[n, n] = self.potential((self.coord[vrhovi[0]] + self.coord[vn]) / 2)
        self.K[en, en] = self.potential((self.coord[vrhovi[1]] 
                                            + self.coord[vn]) / 2)
        self.E.eliminate_zeros()
        
    def coarsen(self, n: int=0) -> bool:

        self.refill()
        if self.level[n].leaf < 1:
            return False
        if self.level[n].sib.leaf < 1:
            return False
        
        i = self.level[n].sib.n
        if n > i:
            i, n = n, i
        
        w = self.E[:, [n]].indices
        z = self.E[:, [i]].indices
        if w[0] == z[0]:
            v = w[0]
            w = w[1]
            z = z[1]
        elif w[0] == z[1]:
            v = w[0]
            w = w[1]
            z = z[0]
        elif w[1] == z[0]:
            v = w[1]
            w = w[0]
            z = z[1]
        else:
            v = w[1]
            w = w[0]
            z = z[0]
            

        if self.E[min(w, z), n] != 1:
            self.E[min(w, z), n] = 1
        if self.E[max(w, z), n] != -1:
            self.E[max(w, z), n] = -1  
        
        self.W[n, n] *= 2
        self.D[n, n] /= 2
        self.K[n, n] = self.P[v]
        self.level[n] = self.level[n].parent
        if self.level[n].leaf == 0:
            self.level[n].leaf = 1
        self.level[n].n = n
        temp = self.E[:, [n]].indices

        self.S[n] = self.coord[temp[0]] - self.coord[temp[1]]
        self.V[z].remove(i)
        self.V[z].append(n)
        self.E[v, i] = 0
        self.E[z, i] = 0
        self.E[v, n] = 0
        
        self.R.append((v, i))
        self.E.eliminate_zeros()
        self.refill()
        
        return v, i, w, z
    
    def relocate(self):
        
        v, i = self.R.pop()
        vn, en = self.vcnt-1, self.ecnt-1
        self.ecnt -= 1
        self.vcnt -= 1
        
        if v != vn:
            self.coord[v] = self.coord[vn]
            x0 = binary_search(self.E[[vn], :].indptr, 0)
            x1 = binary_search(self.E[[vn], :].indptr, 1)
            if self.E[v, x0] != self.E[vn, x0]:
                self.E[v, x0] = self.E[vn, x0]
            if self.E[v, x1] != self.E[vn, x1]:
                self.E[v, x1] = self.E[vn, x1]
            if self.E[vn, x0] != 0:
                self.E[vn, x0] = 0
            if self.E[vn, x1] != 0:
                self.E[vn, x1] = 0
            self.P[v] = self.P[vn]
            self.V[v] = self.V.pop()
        else:
            self.V.pop()
        if i != en:
            y = self.E[:, [en]].indices
            if self.E[y[0], i] != self.E[y[0], en]:
                self.E[y[0], i] = self.E[y[0], en]
            if self.E[y[1], i] != self.E[y[1], en]:
                self.E[y[1], i] = self.E[y[1], en]
            if self.E[y[0], en] != 0:
                self.E[y[0], en] = 0
            if self.E[y[1], en] != 0:
                self.E[y[1], en] = 0
            self.level[i] = self.level.pop()
            self.level[i].n = i
            self.S[i] = self.S[en]
            self.K[i, i] = self.K[en, en]
            self.D[i, i] = self.D[en, en]
            self.W[i, i] = self.W[en, en]
            self.V[y[0]].remove(en)
            self.V[y[0]].append(i)
            self.V[y[1]].append(i)
            self.V[y[1]].remove(en)
        else:
            self.level.pop()
        self.E.eliminate_zeros()
        
    def refill(self):
        
        while self.R:
            self.relocate()
        
    def calculate(self):

        self.refill()
        Eabs = np.abs(self.E)
        L = (self.E @ self.D @ ((self.E).T))[:self.vcnt, :self.vcnt] 
        M = (Eabs @ (self.K * self.W) @ np.abs(self.E.T))/6
        M += ss.spdiags(((self.P * 
                          (Eabs @ self.W * Eabs).sum(-1))/6), diags=0, m=(self.E.shape[0], 
                                         self.E.shape[0]), format='csc')
        self.M = M[:self.vcnt, :self.vcnt]
        self.H = L + self.M
        f = self.vfun(self.coord[:self.vcnt])
        self.u = ss.linalg.spsolve(self.H, self.M @ f)
        temp = self.u[self.E.indices]
        self.grad = (temp[::2] - temp[1::2])[:, None] / (self.S[:self.ecnt])

    def calculate_eigen(self, k):

        self.refill()
        Eabs = np.abs(self.E)
        L = (self.E @ self.D @ ((self.E).T))[:self.vcnt, :self.vcnt] 
        M = (Eabs @ (self.K * self.W) @ np.abs(self.E.T))/6
        M += ss.spdiags(((self.P * 
                          (Eabs @ self.W * Eabs).sum(-1))/6), diags=0, m=(self.E.shape[0], 
                                         self.E.shape[0]), format='csc')
        self.M = M[:self.vcnt, :self.vcnt]
        self.H = L + self.M
        f = self.vfun(self.coord[:self.vcnt])
        self.u = ss.linalg.eigs(self.H, k=6, M=self.M, sigma=None, which='SM')[1][:, k]
        temp = self.u[self.E.indices]
        self.grad = (temp[::2] - temp[1::2])[:, None] / (self.S[:self.ecnt])
        
    def wrap_random(self):
        
        n = np.random.randint(0, self.ecnt-1)
        col = self.E[:, [n]].indices
        
        while len(col) != 2:
            n -= 1
            col = self.E[:, [n]].indices
            
        return n

    def integrate_difference(self, n: int, event: bool):

        if event:
            vrhovi = self.E[:, [n]].indices
            u = np.hstack((self.u, (self.u[vrhovi[0]]+self.u[vrhovi[1]])/2))
            self.refine(n)
            self.calculate()
            u = self.u - u
            temp = u[self.E.indices].reshape(2, self.ecnt, order='F')
            ba = (temp[0] * temp[1]) >= 0
            vec_arg = np.vstack((temp, self.W.data[:self.ecnt], ba))
            return jnp.sum(jax.vmap(abs_diff_int, in_axes=(1))(vec_arg))
        W = np.copy(self.W.data[:self.ecnt])
        u = np.copy(self.u)
        E = np.copy(self.E.indices)
        value_or_false = self.coarsen(n)
        if not value_or_false:
            return False
        v, i, w, z = value_or_false
        self.calculate()
        u2 = np.copy(self.u)
        try:
            u2 = np.hstack((u2, u2[v]))
            u2[v] = (u2[w] + u2[z])/2
        except IndexError:
            u2 = np.hstack((u2, (u2[w] + u2[z])/2))
        u = u - u2
        temp = u[E].reshape(2, self.ecnt+1, order='F')
        ba = (temp[0] * temp[1]) >= 0
        vec_arg = np.vstack((temp, W, ba))
        return jnp.sum(jax.vmap(abs_diff_int, in_axes=(1))(vec_arg))
 
    def jump(self, location: int):
        vrhovi = self.E.indices[location*2:location*2+2]
        cum = 0
        for i in vrhovi:
            for j in self.V[i]:
                temp1 = self.grad[location]
                temp2 = self.grad[j]
                t1 = np.max(temp1) == np.inf
                t2 = np.max(temp2) == np.inf
                temp1[t1] = 0
                temp1[t2] = 0
                temp2[t1] = 0
                temp2[t2] = 0
                cum += np.linalg.norm(temp1 - temp2)
        return cum

    def neighbor_jump(self, location: int):
        cum = 0
        vrhovi = self.E.indices[location*2:location*2+2]
        for i in vrhovi:
            for j in self.V[i]:
                if j!=location:
                    cum += self.jump(j)
        return cum

    def average_jump(self):
        cum = 0
        for i in range(self.ecnt):
            cum += self.jump(i)
        return cum / self.ecnt
    

class Topology:

    def __init__(self, 
                 graph: np.ndarray = None,
                 coord: np.ndarray = None,
                 fun: Callable[[np.ndarray], float] = lambda x: 1,
                 cond: np.ndarray = None
                 ) -> None:
        self.nnodes = graph.shape[0]
        self.fun = fun
        self.vfun = jax.vmap(fun)
        self.coord = coord
        self.F = self.vfun(self.coord)
        self.size = graph.shape[1]
        self.C = cond
        self.E = ss.csr_array(graph)
        self.G = ss.csc_array(graph)
        self.coord = coord
        self.S = coord[self.G.indices]
        self.B = la.norm(self.S[::2] - self.S[1::2], axis=1)
        self.D = ss.spdiags(self.C/self.B, diags=0, m=(self.E.shape[1], 
                                         self.E.shape[1]), format='csr')
        indptr = self.E.indptr
        indices = self.E.indices
        msgsize = np.diff(indptr)
        self.leafs = msgsize == 1
        self.senders = (graph.shape[1]+1)*np.ones(7*self.size, dtype=int)
        self.receivers = (graph.shape[1]+1)*np.ones(7*self.size, dtype=int)
        cnt = 0

        for i in np.split(indices, indptr[1:-1]):
            l = len(i) * (len(i) - 1)
            tmp = np.meshgrid(i, i)
            a = tmp[0].flatten()
            b = tmp[1].flatten()
            tmp = a!=b
            self.senders[cnt:cnt+l] = a[tmp]
            self.receivers[cnt:cnt+l] = b[tmp]
            cnt += l
        self.senders[-self.size:] = np.arange(self.size)
        self.receivers[-self.size:] = np.arange(self.size)
        self.features = np.zeros((self.E.shape[1]+1, 5))
        self.features[-1, :] = np.zeros((5,))
        self.features[:-1, 0] = self.coord[:, 0][self.G.indices[::2]]
        self.features[:-1, 1] = self.coord[:, 1][self.G.indices[1::2]]
        self.features[:-1, 4] = self.C[:]

    def calculate(self):
        self.L = (self.E @ self.D @ ((self.E).T))[:-1, :-1]
        self.u = np.hstack((ss.linalg.spsolve(self.L, self.F[:-1]), 0))
        self.features[:-1, 2] = self.u[self.G.indices[::2]]
        self.features[:-1, 3] = self.u[self.G.indices[1::2]]
        self.fc = (-self.u @ self.F)/(len(self.F)**2)
        
    def increase(self, n):
        self.D[n] /= self.C[n]
        self.C[n] += 0.1
        self.D[n] *= self.C[n]
        self.features[n, 4] += 0.1

