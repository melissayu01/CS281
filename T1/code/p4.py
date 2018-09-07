import numpy as np

np.set_printoptions(precision=8)

data = np.loadtxt("CASP.csv", delimiter = ",", skiprows = 1)
y, X = data[:, 0], data[:, 1:]
n, d = X.shape
d += 1

# normalization
def normalize_and_add_bias(X):
    out = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return np.insert(out, 0, values=1, axis=1)

# train/test split
n_train = int(0.9 * n)
train_X, test_X = X[:n_train, :], X[n_train:, :]
train_X, test_X = normalize_and_add_bias(train_X), normalize_and_add_bias(test_X)
train_y, test_y = y[:n_train], y[n_train:]

# augment training data
sigma = 1
tau = 1 / np.sqrt(10)
train_X = np.concatenate((train_X / sigma, np.identity(d) / tau), axis = 0)
train_y = np.concatenate((train_y / sigma, np.zeros((d, ))), axis = 0)

# QR decomposition
Q, R = np.linalg.qr(train_X)
w = np.linalg.inv(R) @ Q.T @ train_y

# RMSE
y_hat = test_X @ w
rmse = np.sqrt((np.linalg.norm(test_y - y_hat) ** 2) / (n - n_train))

# results
print('w = {}\nRMSE = {:0.8f}'.format(w, rmse))
