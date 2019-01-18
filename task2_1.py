import mnist
import numpy as np
import one_hot_encoding as ohe
import matplotlib.pyplot as plt



#Loading data
training_data_size = 20000
testing_data_size = 2000

X_train, Y_train, X_test, Y_test = mnist.load()
X_train = X_train/255
X_test = X_test/255
X_train = np.concatenate((X_train,np.ones([60000,1])), axis=1)
X_test = np.concatenate((X_test,np.ones([10000,1])), axis=1)
training_data_output = ohe.one_hot_encoding(Y_train)
training_data_input = X_train[0:training_data_size,:].copy()
testing_data = X_test[-testing_data_size:,:].copy()

#Hypervariables
epoch = 20
learning_rate = 0.001
batch_size = 20

#Data storage
error_vector = []

def error_function(y_n,t_n):
    return np.dot(y_n,t_n)

def feed_forward(w,x):
    return np.dot(w,x)

def gradient_function(x_n,y_n,t_n): #Returns gradient with given testing data
    
    return np.outer((t_n-y_n), x_n)

def minimizing_direction(w,x,t, i):

    gradient = np.zeros([10,785])
    error = 0
    number_of_training_sets = min(batch_size,training_data_size-i)
    if number_of_training_sets == 0: return 0

    for k in range(number_of_training_sets):

        #Finding training data
        x_n = x[i+k]
        y_n = feed_forward(w,x_n)
        t_n = t[i+k]
        error += error_function(y_n,t_n)

        gradient += gradient_function(x_n,y_n,t_n)
    
    gradient = gradient/number_of_training_sets
    error = error/number_of_training_sets

    return (-learning_rate*gradient,error)

def gradient_descent(training_data_input, training_data_output):
    
    w = np.random.rand(10,785)

    for e in range(epoch):
        i = 0
        while i < training_data_size:

            min_dir, error = minimizing_direction(w,training_data_input,training_data_output,i)
            w = w - min_dir
            i += batch_size

        print("Epoch:", e+1, " | Current error: ", error)
        error_vector.append(error)
    
    return w


gradient_descent(training_data_input,training_data_output)
plt.plot(error_vector)
plt.show()