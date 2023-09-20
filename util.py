from const import *

import numpy as np
import jax.numpy as jnp
import jax
from math import gamma
import math
from numpy.linalg import eig
import matplotlib.pyplot as plt

@jax.vmap
def euclidean(x):
    return jnp.sqrt(jnp.sum(x**2))

def barrier(p):
    return np.sqrt(p)/(1-p)

def integrate(x, u):

    p = np.abs(u)
    p = p[:-1, :] + p[1:, :]
    r = np.diff(x, axis=0)
    
    return np.sum(p*r/2)

def JacobiP(x, alpha, beta, N):
        
    if abs(alpha + beta + 1) < 1e-14:
        gamma0 = gamma(alpha + 1) * gamma(beta + 1)
    else:
        gamma0 = 2**(alpha + beta + 1) / (alpha + beta + 1.) \
                 * gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1)

    pl = [1.0 / math.sqrt(gamma0) + 0 * x]

    if N == 0:
        return pl[0]

    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    pl.append(((alpha + beta + 2) * x / 2 + \
               (alpha - beta) / 2) / math.sqrt(gamma1))

    if N == 1:
        return pl[1]

    aold = 2. / (2. + alpha + beta) * math.sqrt((alpha + 1.) * (beta + 1.)
                                                / (alpha + beta + 3.))

    for i in range(1, N):
        h1 = 2. * i + alpha + beta
        foo = (i + 1.) * (i + 1. + alpha + beta) * \
              (i + 1. + alpha) * (i + 1. + beta) / (h1 + 1.) / (h1 + 3.)
        anew = 2. / (h1 + 2.) * math.sqrt(foo)
        bnew = -(alpha * alpha - beta * beta) / (h1 * (h1 + 2.))
        pl.append((-aold * pl[i - 1] + np.multiply(x - bnew, pl[i])) / anew)
        aold = anew

    return pl[N]

def GradJacobiP(x, alpha, beta, N):

    if N == 0:
        return np.zeros((len(x)))
    else:
        return math.sqrt(N * (N + alpha + beta + 1)) * \
               JacobiP(x, alpha + 1, beta + 1, N - 1)

def JacobiGL(alpha, beta, N):
        
    x = np.zeros(N+1,)

    if N == 1:
        x[0] = -1.
        x[1] = 1.
        return x

    xint, w = JacobiGQ(alpha + 1, beta + 1, N - 2)
    x = np.hstack((-1., xint, 1.))

    return x

def JacobiGQ(alpha, beta, N):

    if N == 0:
        x = np.array([(alpha - beta) / (alpha + beta + 2)])
        w = np.array([2])
        return x, w

    J = np.zeros(N + 1)
    h1 = 2 * np.arange(N + 1) + alpha + beta
    temp = np.arange(N) + 1
    J = np.diag(-1/2*(alpha**2-beta**2)/(h1+2)/h1) + \
        np.diag(2/(h1[:N]+2)*np.sqrt(temp*(temp+alpha+beta) * \
        (temp+alpha)*(temp+beta)/(h1[:N]+1)/(h1[:N]+3)), 1)

    if alpha + beta < 10 * np.finfo(float).eps:
            J[0, 0] = 0.

    J = J + J.T
    D, V = eig(J)
    sorter = np.argsort(D)
    D = D[sorter]
    V = V[:, sorter]
    x = D
    w = np.square(V[0, :].T) * 2**(alpha + beta + 1) / ( alpha + beta + 1) * \
        gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1)

    return x, w

def plot_graph(T):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(T.feat[:, 0][:T.vcnt],
            T.feat[:, 1][:T.vcnt],
            T.feat[:, 2][:T.vcnt])
    plt.show()

def plot2d(T):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(T.feat[:, 0][:T.vcnt],
           T.feat[:, 1][:T.vcnt],
           T.u)
    plt.show()

def plot2d_and_graph(T):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(T.feat[:, 0][:T.vcnt],
            T.feat[:, 1][:T.vcnt],
            T.feat[:, 2][:T.vcnt])
    ax.scatter(T.feat[:, 0][:T.vcnt],
           T.feat[:, 1][:T.vcnt],
           T.u)
    plt.show()

def plot3d(T):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(T.feat[:, 0][:T.vcnt],
            T.feat[:, 1][:T.vcnt],
            T.feat[:, 2][:T.vcnt],
            c=T.u, cmap='coolwarm')
    cbar = plt.colorbar(ax.collections[0])
    cbar.set_label('Function Value')
    plt.show()

def binary_search(row, e):
    lo = 0
    hi = len(row) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if row[mid] == e:
            sol = mid
            lo = mid + 1
        elif row[mid] > e:
            hi = mid - 1
        else:
            lo = mid + 1
    return sol

@jax.jit
def abs_diff_int(x):
    return (jnp.abs(x[0]+x[1])/2 * x[2]) * x[3] + \
        jnp.logical_not(x[3]) * \
        ((x[0]**2 + x[1]**2) * x[2] / (jnp.abs(x[0]) + jnp.abs(x[1])) / 2)