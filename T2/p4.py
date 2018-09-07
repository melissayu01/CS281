import numpy as np
import utils

class NaiveBayesClassifier:
    def __init__(self, alpha, beta, n_features):
        # 1 x C vector; dirichlet prior for class distr.
        self.alpha = alpha
        self.alpha0 = sum(alpha)

        # 1 x K vector; dirichlet prior for class conditional distr.
        self.beta = beta
        self.beta0 = sum(beta)

        # dimensions of data
        self.C = len(self.alpha)
        self.K = len(self.beta)
        self.D = n_features

        # counts
        self.N = 0
        self.N_c = np.zeros(self.C, dtype=int)
        self.N_cj = np.zeros((self.C, self.D), dtype=int)
        self.N_ckj = np.zeros((self.C, self.K, self.D), dtype=int)

        self.flushed = False

    def fit(self, X, y):
        X = X.astype(int)
        N, _D = X.shape
        self.N += N

        # print("Fitting model")
        for c in range(self.C):
            msk = y == c
            self.N_c[c] += np.sum(msk)
            self.N_cj[c] += np.sum(X[msk], dtype=int, axis=0)
            self.N_ckj[c] += np.apply_along_axis(np.bincount, 0, X[msk], minlength=self.K)

        self.flushed = False

    def predict(self, X):
        X = X.astype(int)

        if not self.flushed:
            # print("Flushing")
            self.pi = np.array([
                np.log(self.N_c[c] + self.alpha[c]) - np.log(self.N + self.alpha0)
                for c in range(self.C)])
            self.mu = np.fromfunction(
                lambda c, j, k: np.log(self.N_ckj[c, k, j] + self.beta[c]) - np.log(self.N_c[c] + self.beta0),
                (self.C, self.D, self.K), dtype=int
            )
            self.flushed = True

        # print("Predicting labels")
        p_for_x = lambda x: [self.pi[c] + np.sum([self.mu[c, j, x[j]] for j in range(len(x))]) for c in range(self.C)]
        ps = np.apply_along_axis(p_for_x, 1, X)
        return np.apply_along_axis(np.argmax, 1, ps)

def main(b, binary):
    # load data
    train_iter, val_iter, test_iter, text_field = utils.load_imdb(batch_size=1000)

    # initialize classifier
    alpha = np.ones(2)
    beta = b * np.ones(281 + 1)
    n_features = 245703
    nb = NaiveBayesClassifier(alpha, beta, n_features)

    print("training")
    # train
    i = 0
    for batch in train_iter:
        print(i)
        X = utils.bag_of_words(batch, text_field).data.numpy()
        if binary:
            X = X > 0
        y = batch.label.data.numpy() - 1
        nb.fit(X, y)
        i += 1

    print("testing")
    # test
    n, n_corr = 0, 0
    i = 0
    for batch in test_iter:
        X = utils.bag_of_words(batch, text_field).data.numpy()
        if binary:
            X = X > 0
        y_pred = nb.predict(X)
        y = batch.label.data.numpy() - 1

        n += len(y)
        n_corr += sum(y_pred == y)
        i += 1

    return n_corr / n

if __name__ == '__main__':
    for binary in [False]:
        for b in [0.1, 0.5]:
            print(">>> beta = {}, binary features = {}".format(b, binary))
            print(main(b, binary))
