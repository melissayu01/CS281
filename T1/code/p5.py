import numpy as np
import torch

np.set_printoptions(precision=8)

data = np.loadtxt("CASP.csv", delimiter = ",", skiprows = 1)
y, X = data[:, 0], data[:, 1:]
n, d = X.shape

# train/test split
n_train = int(0.9 * n)
train_X, test_X = X[:n_train, :], X[n_train:, :]
train_y, test_y = y[:n_train], y[n_train:]

# normalization
def normalize_and_add_bias(X):
    out = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return np.insert(out, 0, values=1, axis=1)
train_X, test_X = normalize_and_add_bias(train_X), normalize_and_add_bias(test_X)
d += 1

# pytorch weights optimizer
weights = torch.autograd.Variable(torch.Tensor(np.random.randn(d)), requires_grad=True)
optimizer = torch.optim.LBFGS([weights])

# update function
def black_box():
    sigma = 1
    tau = 1 / np.sqrt(10)

    weights_data = weights.data.numpy()

    res = train_y - train_X @ weights_data
    loss = res.T @ res/ (sigma ** 2) + weights_data.T @ weights_data / (tau ** 2)

    gradient = - (train_X.T @ train_y - train_X.T @ train_X @ weights_data) / (sigma ** 2)
    + weights_data / (tau ** 2)

    weights.grad = torch.autograd.Variable(torch.Tensor(gradient))

    return loss

# optimal weights after 100 iters
for _ in range(100):
    optimizer.step(black_box)
w = weights.data.numpy()

# RMSE
y_hat = test_X @ w
rmse = np.sqrt((np.linalg.norm(test_y - y_hat) ** 2) / (n - n_train))

# results
print('w = {}\nRMSE = {:0.8f}'.format(w, rmse))
