#尊重这个行业，勤于练习
#load data module use from Dr. Andrew Ng
#Thanks for Dr. Andrew Ng's teaching
#attation:when you use numpy coding ,don't forget use keepdims=True
#如果使用randn初始化且不使用*0.01操作你的值大概率溢出,检查你的函数注意sigmod计算完要返回值,除了溢出以外还有可能导致梯度消失问题
# 尤其是使用sigmoid时
#记住你算出来的dw是竖向的与你的w的方向是相反的记得转换
#numpy.zeros需要两个括号
#数据堆叠与torch不同采用水平堆叠数据

import numpy as np
import h5py
from lr_utils import load_dataset
import matplotlib.pyplot as plt
import copy
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_flatten =train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten =test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache
def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache
def relu_backward(x):
    s=0
    if(x>0):
        s=1
    else:
        s=0
    return s
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ
def sigmoid_backward(dA, cache):

    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)



    return dZ
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters
def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache
#L层前向传播
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        #除了最后一层都采用relu激活
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    #最后一层采用sigmoid激活
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (np.sum(Y * np.log(AL), axis=1, keepdims=True) + np.sum((1 - Y) * np.log(1 - AL), axis=1, keepdims=True)) / (
        -m)
    cost = np.squeeze(cost)
    return cost
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = dZ.dot(A_prev.T) * (1 / m)
    db = np.sum(dZ, axis=1, keepdims=True) * (1 / m)
    dA_prev = (W.T).dot(dZ)
    return dA_prev, dW, db
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads
def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - grads["dW" + str(l + 1)] * learning_rate
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - grads["db" + str(l + 1)] * learning_rate
    return parameters
def L_layer_model(X, Y, layers_dims, learning_rate=0.005, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs
def accuracy(predic,true):
    pred=copy.deepcopy(predic)
    true=copy.deepcopy(true)
    pred=np.squeeze(pred)
    pred=np.expand_dims(pred, axis=0)
    true=np.squeeze(true)
    true = np.expand_dims(true, axis=0)
    true_compulate=[]
    for i in range(pred.shape[1]):
        if(pred[0][i]==true[0][i]):
            true_compulate.append(1)

        else:

            true_compulate.append(0)
    accur=np.array(true_compulate)
    accur=str(accur.sum()/len(accur)*100)+"%"

    return accur
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
parameters, costs = L_layer_model(train_set_x, train_set_y, layers_dims, num_iterations = 2500, print_cost = True)
"""ytrain=
ytest=
accuracy_train=accuracy(ytrain,train_set_y)
accuracy_test=accuracy(ytest,test_set_y)
print(f"Accuracy for train_set is {accuracy_train}")
print(f"Accuracy for test_set is {accuracy_test}")"""


