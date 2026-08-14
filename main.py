import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\LENOVO\\Documents\\vscode\\nn scratch\\archive(1)\\mnist_train.csv')
data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

data_dev = data[0:10000] # shape [10000,785]
x_dev = data_dev[:,1:n] #We want only the 784 pixels and not the label at col 0. shape is [10000,784] FK THIS BRO
y_dev = data_dev[:,0] #We want to take the labels at col 0. shape is [10000,1]
dev = (x_dev, y_dev)

data_train = data[10000:m] #shape [50000,785]
x_train = data_train[:,1:n] 
y_train= data_train[:,0] 
train =(x_train,y_train)

def parameters(): 
    w1 = np.random.rand(784,158) - 0.5 
    w2 = np.random.rand(158, 79) - 0.5
    w3 = np.random.rand(79, 10) - 0.5
    b1 = np.random.rand(1, 158) - 0.5
    b2 = np.random.rand(1, 79) - 0.5
    b3 = np.random.rand(1, 10) - 0.5
    return w1, w2, w3, b1 , b2, b3
def ReLU(z): 
    z = np.max(z,0)
    return z
def softmax(z): 
    z = np.exp(z)/sum(np.exp(z))
    return z
def forwardprop(w1, w2, w3, b1, b2 , x): 
    z1 = x.dot(w1) + b1 
    a1 = ReLU(z1)
    z2 = a1.dot(w2) + b2 
    a2 = ReLU(z2)
    z3 = a2.dot(w3) + b2 
    output = softmax(z3)
    return z1, z1, z2, a2, z3 , output
def one_hot_encoding(y): 
    one_hot_y = np.zeros((m,10))
    one_hot_y[np.arange(m),y] = 1 
    return one_hot_y

def ReLU_deriv(z):
    return z>0

def backprop(w2, w3, z1,z2 ,a1 ,a2, output): 
    dz3 = output - one_hot_encoding(y_train)
    dw3 = 1/m * a2.T.dot(dz3)
    db3 = 1/m * np.sum(dz3, axis = 0, keepdims=True)
    dz2 = dz3.dot(w3.T) * ReLU_deriv(z2)
    dw2 = 1/m * a1.T.dot(dz2)
    db2 = 1/m * np.sum(dz2, axis = 0, keepdims = True )
    dz1 = dz2.dot(w2.T)* ReLU_deriv(z1)
    dw1 = 1/m * x_train.T.dot(dz1) 
    db1 = 1/m * np.sum(dz1, axis = 0, keepdims = True)
    return dz3,dw3,db3,dz2,dw2,db2,dz1,dw1,db1
def update_params(w1, w2, w3, b1, b2, b3, dw1,dw2,dw3,db1,db2,db3, learning_rate): 
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    return w1, w2, w3, b1, b2, b3 

def gradient_descent()