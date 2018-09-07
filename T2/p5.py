import utils
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim

def build_model(input_dim, output_dim):
    model = torch.nn.Sequential()
    # computes w_c^T x + b_c
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim))
    # Compute our log softmax term.
    model.add_module("softmax", torch.nn.LogSoftmax())
    return model

def train(model, loss, reg_weight, optimizer, x_val, y_val):
    # Take in x and y and make variable.
    x = Variable(x_val)
    y = Variable(y_val)

    # Resets the gradients to 0
    optimizer.zero_grad()

    # Computes the function above. (log softmax w_c^T x + b_c)
    fx = model.forward(x)

    # Computes loss. Gives a scalar.
    output = loss.forward(fx, y)
    l1_crit = torch.nn.L1Loss(size_average=False)
    target = Variable(torch.zeros(2,245703), requires_grad=False)
    param = next(model.parameters())
    reg_loss = l1_crit(param, target)
    output += reg_weight * reg_loss

    # Magically computes the gradients.
    output.backward()

    # updates the weights
    optimizer.step()
    return output.data[0]

def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)

def main():
    torch.manual_seed(42)
    n_features = 245703
    n_classes = 2
    reg_weights = [0,0.0001,0.001,0.005,0.01,0.1,1]

    for reg_weight in reg_weights:
        train_iter, val_iter, test_iter, text_field = utils.load_imdb(batch_size=1000)

        # build model
        model = build_model(n_features, n_classes)

        # Loss here is negative log-likelihood
        loss = torch.nn.NLLLoss(size_average=True)

        # Optimizer. SGD stochastic gradient.
        optimizer = optim.Adam(model.parameters())

        cost = 0.
        num_batches = 0
        for batch in train_iter:
            X = utils.bag_of_words(batch, text_field).data
            y = batch.label.data - 1
            cost += train(model, loss, reg_weight, optimizer, X, y)
            num_batches += 1

        n, n_corr = 0, 0
        for batch in test_iter:
            X = utils.bag_of_words(batch, text_field).data
            y = batch.label.data - 1
            y_pred = predict(model, X)

            n += len(y.numpy())
            n_corr += sum(y_pred == y.numpy())

        print("Lambda %f, cost = %f, acc = %.2f%%"
              % (reg_weight, cost / num_batches, 100. * n_corr / n))
        weights = next(model.parameters()).data.numpy()
        heaviest = [[text_field.vocab.itos[word_id] for word_id in np.argsort(w)[-5:][::-1]] for w in weights]
        lightest = [[text_field.vocab.itos[word_id] for word_id in np.argsort(w)[:5][::-1]] for w in weights]
        sparsity = np.sum(weights < 1e-4) / (weights.shape[0] * weights.shape[1])
        print(">>> Heaviest Words\nClass 0: {}\nClass 1: {}".format(heaviest[0], heaviest[1]))
        print(">>> Lightest Words\nClass 0: {}\nClass 1: {}".format(lightest[0], lightest[1]))
        print(">>> Sparsity: {}".format(sparsity))
