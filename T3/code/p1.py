"""
1.
"""
import math
import numpy as np
import itertools
from functools import reduce
import operator
import torch
from torch.autograd import Variable
torch.set_printoptions(precision=4)

class P1:
    def __init__(self):
        self.dtype = torch.FloatTensor
        self.torch_of = lambda y: Variable(torch.Tensor(y).type(self.dtype), requires_grad=False)

        self.names = ["SR", "MG", "RS", "YD", "ZH", "HS", "NZ", "YK"]
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.n = len(self.names)

        # useful helper functions
        self.name_of = lambda i: self.names[i]
        self.index_of = lambda n: self.indices[n]
        self.all_ys = lambda: map(self.torch_of, itertools.product(range(2), repeat=self.n))

        # parameters of model
        self.forward_edges = [(self.index_of(s), self.index_of(t)) for (s, t) in
                 [("MG", "RS"), ("RS", "YD"), ("RS", "ZH"),
                 ("ZH", "YK"), ("ZH", "HS"), ("ZH", "NZ")]
                ]
        self.backward_edges = [(j, i) for (i, j) in self.forward_edges][::-1]
        self.messages = dict()

        self.theta_st = 2
        self.theta_s = Variable(
            torch.Tensor([2, -2, 3, -2, -8, -2, -2, 1]).type(self.dtype),
            requires_grad=True
        )

        self.log_partition_cache = None

    def log_potential(self, y):
        edge_scores = (y[s] * self.theta_st * y[t] for s, t in self.forward_edges)
        node_scores = self.theta_s.dot(y)
        return sum(edge_scores) + node_scores

    def log_partition(self):
        if self.log_partition_cache is None:
            scores = [self.log_potential(y).exp() for y in self.all_ys()]
            self.log_partition_cache = sum(scores).log()
        return self.log_partition_cache

    def marginals(self):
        ps = lambda i: sum(((self.log_potential(y) - self.log_partition()).exp()
              for y in self.all_ys() if y[i].data[0] == 1)).data[0]
        return self.torch_of([ps(i) for i in range(self.n)])

    def marginals_optim(self):
        if self.theta_s.grad is None:
            lpf = self.log_partition()
            lpf.backward()
        return self.theta_s.grad

    def marginals_bp(self):
        prod = lambda l: reduce(operator.mul, l, 1)

        psi_s = lambda s, ys: (self.theta_s[s] * ys).exp()
        psi_st = lambda ys, yt: self.torch_of([ys * self.theta_st * yt]).exp()

        children = lambda s: [c for (p, c) in self.forward_edges if p == s]
        parents = lambda t: [p for (p, c) in self.forward_edges if c == t]
        neighbors = lambda i: children(i) + parents(i)

        # backward pass
        for (s, t) in self.backward_edges:
            for yt in range(2):
                ms = [
                    psi_st(ys, yt) * psi_s(s, ys) * prod((self.messages[(c, s, ys)] for c in children(s)))
                    for ys in range(2)
                ]
                self.messages[(s, t, yt)] = sum(ms)

        # forward pass
        for (t, s) in self.forward_edges:
            for ys in range(2):
                ms = [
                    psi_st(ys, yt) * psi_s(t, yt)
                    * prod((self.messages[(c, t, yt)] for c in children(t) if c != s))
                    * prod((self.messages[(p, t, yt)] for p in parents(t)))
                    for yt in range(2)
                ]
                self.messages[(t, s, ys)] = sum(ms)

        # marginals
        belief = lambda i, y: psi_s(i, y) * prod((self.messages[(n, i, y)] for n in neighbors(i)))
        p = []
        for i in range(self.n):
            p0 = belief(i, 0)
            p1 = belief(i, 1)
            p.append(p1 / (p0 + p1))

        return p

    def get_test_y(self):
        return self.torch_of([
                int(i == self.index_of("RS") or i == self.index_of("SR"))
                for i in range(self.n)
            ])

p1 = P1()
y = p1.get_test_y()

print(p1.log_potential(y))
print(p1.log_partition())
print(p1.marginals(), p1.marginals_optim(), p1.marginals_bp())
