import numpy as np
import h5py
from lr_utils import load_dataset
import matplotlib.pyplot as plt
import copy
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid((w).dot(X) + b)
    cost = (np.sum(Y * np.log(A), axis=1, keepdims=True) + np.sum((1 - Y) * np.log(1 - A), axis=1, keepdims=True)) * (
                -1 / m)
    dw = (1 / m) * ((X * (A - Y)).sum(axis=1, keepdims=True))
    db = np.float64((1 / m) * (np.sum((A - Y), axis=1, keepdims=True)))
    # YOUR CODE ENDS HERE
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}
    return grads, cost
w=np.random.randn(1,12288)*0.1
b=0.0
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_flatten =train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten =test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten / 1.
test_set_x = test_set_x_flatten / 1.
def optimize(w, b, X, Y, num_iterations=100000, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        # (≈ 1 lines of code)
        # Cost and gradient calculation
        # grads, cost = ...
        # YOUR CODE STARTS HERE
        grads, cost = propagate(w, b, X, Y)

        # YOUR CODE ENDS HERE

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        # w = ...
        # b = ...
        # YOUR CODE STARTS HERE
        w = w - learning_rate * (dw.T)
        b = b - learning_rate * (db.T)

        # YOUR CODE ENDS HERE

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs
W,B,C=optimize(w,b,train_set_x,train_set_y,print_cost=True)
