"""
2.
"""
import utils
import numpy as np

import torch
from torch.autograd import Variable
torch.set_printoptions(precision=4)

import seaborn
import matplotlib.pyplot as plt

class LatentLinearModel(torch.nn.Module):
    def __init__(self, K, N, J):
        super(LatentLinearModel, self).__init__()
        self.K = K
        self.U = torch.nn.Embedding(N, K)
        self.V = torch.nn.Embedding(J, K)
        self.a = torch.nn.Embedding(N, 1)
        self.b = torch.nn.Embedding(J, 1)
        self.g = Variable(torch.Tensor(1,), requires_grad=True)

    def forward(self, users, jokes):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        batch_size = self.a(users).size()
        u = self.U(users).unsqueeze(1)
        v = self.V(jokes).unsqueeze(2)
        r = torch.bmm(u, v).squeeze() + self.a(users).squeeze() + self.b(jokes).squeeze() + self.g
        return r

train_RMSEs = []
val_RMSEs = []
Ks = range(2, 3)

for K in Ks:
    # Load data iterators
    train_iter, val_iter, test_iter = utils.load_jester(
        batch_size=100, subsample_rate=1.0, load_text=False)

    # Construct our model by instantiating the class defined above
    N, J = 70000, 150
    model = LatentLinearModel(K, N, J)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Optimize model parameters
    num_epochs = 50
    for epoch in range(num_epochs):
        loss_for_epoch = 0
        n_in_epoch = 0

        train_iter.init_epoch()
        for batch in train_iter:
            # 0-index data
            ratings = (batch.ratings-1).float()
            users = batch.users-1
            jokes = batch.jokes-1
            n_in_epoch += ratings.size()[0]

            # Forward pass: Compute predicted y by passing x to the model
            r_pred = model(users, jokes)

            # Compute and print loss
            loss = criterion(r_pred, ratings)
            loss_for_epoch += loss.data[0]

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, loss_for_epoch / n_in_epoch)

    train_RMSEs.append(loss_for_epoch / n_in_epoch)

    # Predict on validation set
    val_loss = 0
    n = 0
    for batch in val_iter:
        # 0-index data
        ratings = (batch.ratings-1).float()
        users = batch.users-1
        jokes = batch.jokes-1
        n += ratings.size()[0]

        # Forward pass: Compute predicted y by passing x to the model
        r_pred = model(users, jokes)

        # Compute and print loss
        loss = criterion(r_pred, ratings)
        val_loss += loss.data[0]

    val_RMSEs.append(val_loss / n)

    b = model.b.weight.data.numpy()
    best, worst = np.argmax(b), np.argmin(b)
    print(best, b[best])
    print(worst, b[worst])

    print(model.g)


# print(train_RMSEs, val_RMSEs)
# plt.figure()
# plt.plot(Ks, train_RMSEs, c='r', label='Training RMSE')
# plt.plot(Ks, val_RMSEs, c='g', label='Validation RMSE')
# plt.xlabel("K")
# plt.ylabel("RMSE")
# plt.legend(loc='best')
# plt.show()
