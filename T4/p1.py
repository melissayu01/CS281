"""
1.
"""
from scipy.special import logsumexp
import numpy as np
import itertools
import torch
import pickle

import matplotlib
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context("talk")
torch.set_printoptions(precision=4)
np.set_printoptions(precision=4)

class P1:
    def __init__(self, fname):
        self.x = np.load(fname)
        self.N, self.M, self.K = self.x.shape

        # log potentials
        def edge_func(i, j):
            if i == j:
                return 10
            elif abs(i - j) == 1:
                return 2
            else:
                return 0
        self.edge_theta = np.fromfunction(np.vectorize(edge_func), (self.K, self.K))
        self.node_theta = 10 * self.x

        # MF
        mu = np.zeros(self.x.shape) # mu[i, j, k] = q_ij(y_ij = k)

        # LBP
        msgs = None
        prev_msgs = None
        bels = None

    ### helper functions

    def softmax(self, x):
        # normalizes vector x
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def neighbors(self, i, j):
        out = []
        for (di, dj) in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
            i_, j_ = i + di, j + dj
            if 0 <= i_ < self.N and 0 <= j_ < self.M and (i_, j_) != (i, j):
                out.append((i_, j_))
        return out

    def nodes(self):
        return itertools.product(range(self.N), range(self.M))

    def dir_edges(self):
        for t in self.nodes():
            for n in self.neighbors(*t):
                yield (t, n)

    def from_one_hot(self, one_hot_vec):
        return np.where(one_hot_vec)[0][0]

    ### inference

    def brute_force(self):
        def unnormed_log_joint(y):
            p = sum([self.node_theta[t].dot(y[t]) for t in self.nodes()])
            p += 0.5 * sum((
                self.edge_theta[self.from_one_hot(y[n]), self.from_one_hot(y[t])]
                for (n, t) in self.dir_edges()
            ))
            return p

        def all_ys(key=None):
            tuple_to_mat = lambda t: np.asarray(t).reshape((self.N, self.M))
            to_one_hot = lambda a: (a[...,None] == np.arange(self.K)).astype(float)
            arrays = map(tuple_to_mat, list(itertools.product(range(self.K), repeat=self.N*self.M)))
            sorted_arrays = sorted(map(to_one_hot, arrays), key=key)
            return itertools.groupby(sorted_arrays, key=key)

        marginals = np.zeros(self.x.shape)
        for t in self.nodes():
            ps = []
            for k, group in all_ys( key=lambda img: self.from_one_hot(img[t]) ):
                p = logsumexp([ unnormed_log_joint(y) for y in group ])
                ps.append(p)
            marginals[t] = self.softmax(ps) # np.exp(ps - logsumexp(ps))
        return marginals

    def mean_field(self, epochs=30):
        history = [self.x]
        prev_mu = self.softmax(np.zeros(self.x.shape))
        for epoch in range(epochs):
            mu = np.zeros(self.x.shape)
            for (i, j) in self.nodes():
                ngh = sum(( prev_mu[t] for t in self.neighbors(i, j) ))
                mu[i, j] = self.softmax(self.node_theta[i, j] + self.edge_theta.dot(ngh))
            prev_mu = mu
            history.append(prev_mu)
        return history

    def loopy_bp(self, epochs=30):
        history = [self.x]
        prev_msgs = {edge: np.zeros(self.K) for edge in self.dir_edges()}
        bels = np.ones(self.x.shape)
        for epoch in range(epochs):
            msgs = {edge: np.zeros(self.K) for edge in self.dir_edges()}
            for ((i, j), t) in self.dir_edges():
                for k in range(self.K):
                    ngh = lambda l: [
                        prev_msgs[(u, (i, j))][l]
                        for u in self.neighbors(i, j) if u != t
                    ]
                    values = [
                        self.node_theta[i, j, l] + self.edge_theta[k, l] + sum(ngh(l))
                        for l in range(self.K)
                    ]
                    msgs[((i, j), t)][k] = logsumexp(values)
            for s in self.nodes():
                ngh = (msgs[(t, s)] for t in self.neighbors(*s))
                bels[s] = self.softmax(self.node_theta[s] + sum(ngh))
            prev_msgs = msgs
            history.append(bels)
        return history

# print('small')
# p1 = P1('small.npy')
# marginals = p1.brute_force()
# print(marginals)
# plt.figure()
# plt.imshow(marginals)
epochs = 60
p1 = P1('bullseye.npy')
for f in (p1.mean_field,):
    history = f()
    plt.figure()
    plt.imshow(history[0])
    plt.show()
    plt.figure()
    plt.imshow(history[-1])
    plt.show()
    with open('test02.in', 'wb') as f:
        pickle.dump([history[0], epochs], f)
    with open('test02.out', 'wb') as f:
        pickle.dump([history[-1], epochs], f)
