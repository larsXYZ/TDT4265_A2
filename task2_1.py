import mnist
import numpy as np
import one_hot_encoding as ohe



#Loading data
training_data_size = 100#20000
testing_data_size = 10#2000

X_train, Y_train, X_test, Y_test = mnist.load()
X_train = np.concatenate((X_train,np.ones([60000,1])), axis=1)
X_test = np.concatenate((X_test,np.ones([10000,1])), axis=1)
training_data_output = ohe.one_hot_encoding(Y_train)
training_data_input = X_train[0:training_data_size,:].copy()
testing_data = X_test[-testing_data_size:,:].copy()

#Hypervariables
epoch = 20
learning_rate = 0.001
batch_size = 10

def feed_forward(w,x):
    return 

def grad(x_n,y_n,t_n): #Returns gradient with given testing data
    grad = np.zeros(10,785)
    for i in range(batch_size):
        grad = grad + np.transpose(x_n)*(t_n-y_n)
    grad = grad/batch_size
    return grad

def minimizing_direction(x_n,y_n,t_n):
    return -learning_rate*grad(x_n,y_n,t_n)

def gradient_descent(w, training_data_input, training_data_output):
    
    for i in range(epoch):
        w = w + minimizing_direction()


#Network
w = np.random.rand(10,785)

