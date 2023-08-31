#load data module use from Dr. Andrew Ng
#Thanks for Dr. Andrew Ng's teaching
#attation:when you use numpy coding ,don't forget use keepdims=True
import numpy as np
import h5py
from lr_utils import load_dataset
import matplotlib.pyplot as plt
import copy


def show_picture(index):

    plt.imshow(train_set_x_orig[index])
    print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
        "utf-8") + "' picture.")
    plt.show()
def flatten(train_x,test_x):
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # 将图片展开在此处和吴恩达教授一样我们横向堆叠数据
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    return train_set_x,test_set_x

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def propagate(w, b, X, Y):
    m=X.shape[1]

    A = sigmoid((w).dot(X))
    cost = (Y.dot((A.T))+(1-Y).dot(np.log((1-A).T)))*(1/m)

    # dw = (1 / m) * np.dot(train_set, ((A - train_set_y).T))
    #dw = train_set_x.dot((A - Y).T) * (1 / m)
    #db = ((A - Y).sum()) / m
    dw = (1 / m) * ((X * (A - Y)).sum(axis=1, keepdims=True))
    db = np.float64((1 / m) * (np.sum((A - Y), axis=1, keepdims=True)))
    grads={"dw":dw,"db":db}
    return grads,cost
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

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
    return params,costs
"""def predict(w,b,X):
    predict_Y=[]
    a=sigmoid(w.dot(X)+b)
    for i in range(a.shape[1]):
        if(a[i]>0.5):
            predict_Y[0,i]=1
        else:
            predict_Y[0,i]=0
    return predict_Y"""
"""def test_accuracy(w,b,X,Y):
    predicts=predict(w,b,X)
    accuracy=[]
    for i in range(predicts.shape[1]):
        if(predicts[i]==Y[i]):
            accuracy[i]=1
        else:
            accuracy[i]=0
    acc=accuracy.sum(axis=1)
    acc=float(acc/len(predicts))
    return acc"""





#def model(w,b,X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
#    grads,costs=propagate(w,b,X_train,Y_train)
#    params,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
#   return params

"""def train(w,b,X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    param=model(w,b,X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=True)
    w=param["w"]
    b=param["b"]
    #trainacc=test_accuracy(w,b,X_train,Y_train)
    #testacc=test_accuracy(w,b,X_test,Y_test)
    return param#(,trainacc,testacc)"""
w=np.random.randn(1,12288)
b=0.0
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_flatten =train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten =test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.
optimize(w,b,train_set_x,train_set_y,print_cost=True)


