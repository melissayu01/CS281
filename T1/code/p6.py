import numpy as np
import time
import torch

np.set_printoptions(precision=8)

# data = np.loadtxt("CASP.csv", delimiter = ",", skiprows = 1)
# y, X = data[:, 0], data[:, 1:]
# n, d = X.shape

# # train/test split
# n_train = int(0.9 * n)
# train_X, test_X = X[:n_train, :], X[n_train:, :]
# train_y, test_y = y[:n_train], y[n_train:]

# # normalize
# def normalize_and_add_bias(X):
#     out = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
#     return np.insert(out, 0, values=1, axis=1)
# train_X, test_X = normalize_and_add_bias(train_X), normalize_and_add_bias(test_X)
# d += 1

# def time_function(f, *args):
#     time_start = time.time()
#     f(*args)
#     running_time = time.time() - time_start
#     print('{:0.2f}'.format(running_time))
#     return running_time

# qr_rmse = []
# def ridge_qr(train_X, train_y, test_X, test_y, basis):
#     global qr_rmse

#     sigma = 1
#     tau = 1 / np.sqrt(10)

#     # basis function
#     train_X = basis(train_X)
#     test_X = basis(test_X)
#     d = train_X.shape[1]

#     # augment training data
#     train_X = np.concatenate((train_X / sigma, np.identity(d) / tau), axis = 0)
#     train_y = np.concatenate((train_y / sigma, np.zeros((d, ))), axis = 0)

#     # QR decomposition
#     Q, R = np.linalg.qr(train_X)
#     w = np.linalg.inv(R) @ Q.T @ train_y

#     # RMSE
#     y_hat = test_X @ w
#     rmse = np.sqrt((np.linalg.norm(test_y - y_hat) ** 2) / (n - n_train))
#     qr_rmse.append(rmse)

#     # results
#     print('>>> RIDGE, d = {}, RMSE = {:0.8f}, time = '.format(d, rmse), end='')

# bfgs_rmse = []
# def l_bfgs(train_X, train_y, test_X, test_y, basis):
#     global bfgs_rmse

#     # basis function
#     train_X = basis(train_X)
#     test_X = basis(test_X)
#     d = train_X.shape[1]

#     # pytorch weights optimizer
#     weights = torch.autograd.Variable(torch.Tensor(np.zeros(d)), requires_grad=True)
#     optimizer = torch.optim.LBFGS([weights])

#     # update function
#     def black_box():
#         sigma = 1
#         tau = 1 / np.sqrt(10)

#         weights_data = weights.data.numpy()
#         res = train_y - train_X @ weights_data
#         loss = res.T @ res/ (sigma ** 2) + weights_data.T @ weights_data / (tau ** 2)

#         gradient = - (train_X.T @ train_y - train_X.T @ train_X @ weights_data) / (sigma ** 2)
#         + weights_data / (tau ** 2)
#         weights.grad = torch.autograd.Variable(torch.Tensor(gradient))

#         return loss

#     # optimal weights after 100 iters
#     for _ in range(100):
#         optimizer.step(black_box)
#         w = weights.data.numpy()

#     # RMSE
#     y_hat = test_X @ w
#     rmse = np.sqrt((np.linalg.norm(test_y - y_hat) ** 2) / (n - n_train))
#     bfgs_rmse.append(rmse)

#     # results
#     print('>>> L-BFGS, d = {}, RMSE = {:0.8f}, time = '.format(d, rmse), end='')

# def generate_nonlinear_func(orig_dims, dims):
#     A = np.random.multivariate_normal(np.zeros(orig_dims), np.eye(orig_dims), size=(dims))
#     b = np.random.uniform(0, 2 * np.pi, size=dims)
#     def f(X):
#         return np.apply_along_axis(lambda x: np.cos(A @ x + b), 1, X)
#     return f

# time_function(ridge_qr, train_X, train_y, test_X, test_y, lambda X: X)
# time_function(l_bfgs, train_X, train_y, test_X, test_y, lambda X: X)

# qr_time = []
# bfgs_time = []
# for dims in [100, 200, 400, 600]:
#     basis = generate_nonlinear_func(d, dims)
#     qr_time.append(time_function(ridge_qr, train_X, train_y, test_X, test_y, basis))
#     bfgs_time.append(time_function(l_bfgs, train_X, train_y, test_X, test_y, basis))
#     print('')

# print()
# import seaborn
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(qr_time, qr_rmse, c='r', label='QR')
# plt.scatter(bfgs_time, bfgs_rmse, c='b', label='L-BFGS')
# plt.xlabel("time")
# plt.ylabel("RMSE")
# plt.legend(loc='best')
# plt.show()

import seaborn
import matplotlib.pyplot as plt
plt.figure()
plt.scatter([0.73, 1.41, 3.33, 7.55], [5.47932422, 4.97588791, 4.65924294, 4.55947980], c='r', label='QR')
plt.scatter([3.89, 24.04, 173.30, 501.88], [5.47900917, 4.97657630, 4.65974174, 4.56720636], c='b', label='L-BFGS')
plt.xlabel("time")
plt.ylabel("RMSE")
plt.legend(loc='best')
plt.show()
