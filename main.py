import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('MNIST Dataset/mnist_train.csv')
data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

data_dev = data[0:10000] # shape [10000,785]
x_dev = data_dev[:,1:n] / 255.0#We want only the 784 pixels and not the label at col 0. shape is [10000,784] FK THIS BRO
y_dev = data_dev[:,0] #We want to take the labels at col 0. shape is [10000,1]
dev = (x_dev, y_dev)

data_train = data[10000:m] #shape [50000,785]
x_train = data_train[:,1:n] / 255.0 #shape [50000,784]
y_train= data_train[:,0] #shape [50000,1]
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
    a = np.maximum(z,0)
    return a
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    z= exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return z    
def forwardprop(w1, w2, w3, b1, b2, b3, x): 
    z1 = x.dot(w1) + b1 
    a1 = ReLU(z1)
    z2 = a1.dot(w2) + b2 
    a2 = ReLU(z2)
    z3 = a2.dot(w3) + b3 
    output = softmax(z3)
    return z1, a1, z2, a2 , output
def one_hot_encoding(y): 
    one_hot_y = np.zeros((y.size,10))
    one_hot_y[np.arange(y.size),y] = 1 
    return one_hot_y

def ReLU_deriv(z):
    return z>0

def backprop(x, y, w2, w3, z1,z2 ,a1 ,a2, output): 
    m = x.shape[0]
    dz3 = output - one_hot_encoding(y)
    dw3 = 1/m * a2.T.dot(dz3)
    db3 = 1/m * np.sum(dz3, axis = 0, keepdims=True)
    dz2 = dz3.dot(w3.T) * ReLU_deriv(z2)
    dw2 = 1/m * a1.T.dot(dz2)
    db2 = 1/m * np.sum(dz2, axis = 0, keepdims = True )
    dz1 = dz2.dot(w2.T)* ReLU_deriv(z1)
    dw1 = 1/m * x.T.dot(dz1) 
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
def get_accuracy(output, y): 
    output = np.argmax(output, axis=1)
    accuracy = np.mean(output == y)
    return accuracy

def gradient_descent(x,y, iterations, learning_rate): 
    w1, w2, w3, b1 , b2, b3 = parameters()
    for i in range(iterations):
        z1, a1, z2, a2 , output = forwardprop(w1, w2, w3, b1, b2 ,b3 , x)
        dz3,dw3,db3,dz2,dw2,db2,dz1,dw1,db1 = backprop(x, y, w2, w3, z1,z2 ,a1 ,a2, output)
        w1, w2, w3, b1, b2, b3 = update_params(w1, w2, w3, b1, b2, b3, dw1,dw2,dw3,db1,db2,db3, learning_rate)
        print("Current iteration = " , i , "\n" , "accuracy: ", get_accuracy(output, y))
    return w1, w2, w3, b1, b2, b3

w1, w2, w3, b1, b2, b3 = gradient_descent(x_train, y_train, iterations=200, learning_rate=0.1)

    