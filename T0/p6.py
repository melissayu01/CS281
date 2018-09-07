from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context("talk")

# gradient calculation functions
def f(x):
  	return np.cos(x) + x ** 2 + np.exp(x)

def grad_f(x):
  	return - np.sin(x) + 2 * x + np.exp(x)

def grad_check(x, epsilon):
  	return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

# plotting
x_range = np.arange(10)
epsilons = [0.3*1e-4, 0.6*1e-4, 1e-4]

plt.figure()
for eps in epsilons:
    curr_series = []
    for x in x_range:
        curr_series.append(grad_check(x, eps) - grad_f(x))
    plt.plot(x_range, curr_series,
             c=np.random.rand(3,1),
             label='eps = ' + str(eps))
plt.xlabel("x")
plt.ylabel("Error")
plt.legend(loc='best')
plt.show()
