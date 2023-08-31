#尊重这个行业，勤于练习
#load data module use from Dr. Andrew Ng
#Thanks for Dr. Andrew Ng's teaching
#attation:when you use numpy coding ,don't forget use keepdims=True
#如果使用randn初始化且不使用*0.01操作你的值大概率溢出,检查你的函数注意sigmod计算完要返回值
#记住你算出来的dw是竖向的与你的w的方向是相反的记得转换
#numpy.zeros需要两个括号
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
def sigmoid(x):
    s=1.0/(1+np.exp(-x))
    return s
def progate(w,b,X,Y):
    m=X.shape[1]
    A=sigmoid(w.dot(X)+b)
    cost=(Y.dot(np.log((A.T)))+(1-Y).dot(np.log((1-A).T)))*(-1/m)
    cost=np.squeeze(cost)
    dw=np.sum(X.dot((A-Y).T),axis=1,keepdims=True)*(1/m)
    db=np.sum(A-Y,axis=1,keepdims=True)*(1/m)
    grad={"dw":dw,"db":db}
    return grad,cost

def optim(w,b,X,Y,lr=0.005,iter_number=500,print_=True):
    allcost=[]
    w=copy.deepcopy(w)
    b=copy.deepcopy(b)
    for i in range(iter_number):
        grad,cost=progate(w,b,X,Y)
        dw=grad["dw"]
        db=grad["db"]
        w=w-(dw.T)*lr
        b=b-db*lr
        if(i%100==0):
            allcost.append(cost)
            if(print_):
                print(cost)

    param={"w":w,"b":b}
    return param,allcost
"""预测函数"""
def predic(w,b,X):
    a=sigmoid(w.dot(X)+b)
    m=X.shape[1]
    Y_prediction=np.zeros((1, m))
    for i in range (a.shape[1]):
        if(a[0,i]>0.5):
            Y_prediction[0, i] = 1
    else:
        Y_prediction[0, i] = 0
    return Y_prediction



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
w=np.zeros((1,12288))
b=0.0
para,cost=optim(w,b,train_set_x,train_set_y,0.005,4000)
w=para["w"]
b=para["b"]
ytrain=predic(w,b,train_set_x)
ytest=predic(w,b,test_set_x)
accuracy_train=accuracy(ytrain,train_set_y)
accuracy_test=accuracy(ytest,test_set_y)
print(f"Accuracy for train_set is {accuracy_train}")
print(f"Accuracy for test_set is {accuracy_test}")


