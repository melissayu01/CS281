"""
2.
"""
import utils
import numpy as np
import math

import torch
from torch.autograd import Variable
torch.set_printoptions(precision=4)

import seaborn
import matplotlib.pyplot as plt

class LatentLinearModel(torch.nn.Module):
    def __init__(self, K, N, J):
        super(LatentLinearModel, self).__init__()
        self.K = K
        self.N = N
        self.J = J

        self.mu_U = torch.nn.Embedding(N, K)
        self.logvar_U = torch.nn.Embedding(N, K)
        self.logvar_U.weight.data.fill_(-10)

        self.mu_V = torch.nn.Embedding(J, K)
        self.logvar_V = torch.nn.Embedding(J, K)
        self.logvar_V.weight.data.fill_(-10)

    def forward(self, users, jokes):
        batch_size = users.size()[0]
        z_U = Variable(torch.from_numpy(np.random.normal(0, 1, (batch_size, self.K))).float(), requires_grad=False)
        z_V = Variable(torch.from_numpy(np.random.normal(0, 1, (batch_size, self.K))).float(), requires_grad=False)

        u = ( z_U.mul(self.logvar_U(users).exp().sqrt()).add(self.mu_U(users)) ).unsqueeze(1)
        v = ( z_V.mul(self.logvar_V(jokes).exp().sqrt()).add(self.mu_V(jokes)) ).unsqueeze(2)
        r = torch.bmm(u, v).squeeze()
        return r

def vi_loss(model, users, jokes):
    batch_size = users.size()[0]
    var_for_prior = Variable(
        torch.from_numpy(5 * np.ones((batch_size, model.K))).float(),
        requires_grad=False
    )
    loss = (0.5 * (var_for_prior.sqrt() - model.logvar_U(users))
        + (model.logvar_U(users).exp() + model.mu_U(users) ** 2) / 10).sum()
    loss += (0.5 * (var_for_prior.sqrt() - model.logvar_V(jokes))
        + (model.logvar_V(jokes).exp() + model.mu_V(jokes) ** 2) / 10).sum()

    return loss

Ks = range(10, 11)

for K in Ks:
    train_logprobs = []
    train_lower_bound = []
    test_logprobs = []
    test_loss = []

    # Load data iterators
    train_iter, val_iter, test_iter = utils.load_jester(
        batch_size=100, subsample_rate=0.1, load_text=False)

    # Construct our model by instantiating the class defined above
    N, J = 70000, 150
    model = LatentLinearModel(K, N, J)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)

    # Optimize model parameters
    num_epochs = 10
    for epoch in range(num_epochs):
        lower_bound = 0
        logprob_for_epoch = 0
        n_in_epoch = 0

        # sample 100 u_i, v_j to use in loss for epoch
        U_sample = model.mu_U
        V_sample = model.mu_V

        train_iter.init_epoch()
        for batch in train_iter:
            # 0-index data
            ratings = (batch.ratings-1).float()
            users = batch.users-1
            jokes = batch.jokes-1

            n_in_epoch += ratings.size()[0]

            # Forward pass: Compute predicted y by passing x to the model
            r_pred = model(users, jokes)

            # Compute loss
            loss = 0.5 * criterion(r_pred, ratings)
            loss += vi_loss(model, users, jokes) # add entropy and prior terms
            lower_bound -= loss.data[0]

            # Update log-likelihood for epoch
            _r_pred = torch.bmm((U_sample(users)).unsqueeze(1), (V_sample(jokes)).unsqueeze(2)).squeeze()
            _logprob = - 0.5 * criterion(_r_pred, ratings) - ratings.size()[0] * np.log(1/(2 * math.pi))
            logprob_for_epoch += _logprob.data[0]

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lower_bound /= n_in_epoch
        logprob_for_epoch /= n_in_epoch
        train_logprobs.append(logprob_for_epoch)
        train_lower_bound.append(lower_bound)

        # Predict on test set
        test_logprob_for_epoch = 0
        n = 0
        for batch in test_iter:
            # 0-index data
            ratings = (batch.ratings-1).float()
            users = batch.users-1
            jokes = batch.jokes-1

            n += ratings.size()[0]

            # Forward pass: Compute predicted y by passing x to the model
            r_pred = model(users, jokes)

            # Update log-likelihood for epoch
            _r_pred = torch.bmm((U_sample(users)).unsqueeze(1), (V_sample(jokes)).unsqueeze(2)).squeeze()
            _logprob = - 0.5 * criterion(_r_pred, ratings) - ratings.size()[0] * np.log(1/(2 * math.pi))
            test_logprob_for_epoch += _logprob.data[0]

        test_logprob_for_epoch /= n
        test_logprobs.append(test_logprob_for_epoch)

        # Print summary for epoch
        print(epoch, logprob_for_epoch, lower_bound)

    plt.figure()
    plt.plot(range(num_epochs), train_logprobs, c='r')
    plt.xlabel("Epoch")
    plt.ylabel('Training Log-likelihood')

    plt.figure()
    plt.plot(range(num_epochs), test_logprobs, c='b')
    plt.xlabel("Epoch")
    plt.ylabel('Testing Log-likelihood')

    plt.figure()
    plt.plot(range(num_epochs), train_lower_bound, c='g')
    plt.xlabel("Epoch")
    plt.ylabel('Training Lower Bound')

    plt.show()
