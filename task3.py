import mnist
import numpy as np
import one_hot_encoding as ohe
import matplotlib.pyplot as plt

#Data settings
training_data_size = 50000
validation_data_size = 5000
testing_data_size = 1000

#Loading data from MNIST dataset
X_train, Y_train, X_test, Y_test = mnist.load()

#Reducing values to between 0 - 1
X_train = X_train/255
X_test = X_test/255

#Performing the "bias trick"
X_train = np.concatenate((X_train,np.ones([60000,1])), axis=1)
X_test = np.concatenate((X_test,np.ones([10000,1])), axis=1)

#Selecting training data and validation data
training_data_input = X_train[0:training_data_size,:].copy()
training_data_output = Y_train[0:training_data_size].copy()

validation_data_input = X_train[training_data_size:training_data_size+validation_data_size].copy()
validation_data_output = Y_train[training_data_size:training_data_size+validation_data_size].copy()

testing_data_input = X_test[-testing_data_size:].copy()
testing_data_output = Y_test[-testing_data_size:].copy()

#One hot encode the wanted results
training_data_output = ohe.one_hot_encoding(training_data_output)
validation_data_output = ohe.one_hot_encoding(validation_data_output)
testing_data_output = ohe.one_hot_encoding(testing_data_output)

print(np.shape(X_train), np.shape(Y_train))
print(np.shape(X_test), np.shape(Y_test))
input()

#Hypervariables
epoch = 10
batch_size = 10
initial_learning_rate = 0.01 #Initial annealing learning rate
L2_coeffisient = 0.001 #L2 coefficient punishing model complexity, bigger -> less complex weights
T = 300000 #Annealing learning rate time constant, bigger -> slower decrease of learning rate

#Error storage
error_vector_training = []
error_vector_validation = []
error_vector_test = []
percent_correct_training_vector = []
percent_correct_validation_vector = []
percent_correct_testing_vector = []

#Initializing weights 
weights = np.random.rand(10,785)


def update_error_vectors(w):
    #Checking error
    error_training = data_set_test(w,training_data_input,training_data_output)
    error_validation = data_set_test(w,validation_data_input,validation_data_output)
    error_test = data_set_test(w,testing_data_input, testing_data_output)

    #Checking percentage correct
    percent_correct_training_vector.append(percent_correct_test(w, training_data_input, training_data_output))
    percent_correct_validation_vector.append(percent_correct_test(w, validation_data_input, validation_data_output))
    percent_correct_testing_vector.append(percent_correct_test(w, testing_data_input, testing_data_output))

    #Updating storage
    error_vector_training.append(error_training/training_data_size)
    error_vector_validation.append(error_validation/validation_data_size)
    error_vector_test.append(error_test/testing_data_size)

def learning_rate(i, e): #Annealing learning rate
    return initial_learning_rate/(1+(i+training_data_size*e)/T)

def g(y_n):
    return 1/(1+np.exp(-y_n))

def percent_correct_test(w,input_data,output_data):
    
    #Input check
    data_length = np.shape(input_data)[0]
    if np.shape(output_data)[0] != data_length:
        print("Input data shape does not equal output data shape")
        print(np.shape(input_data)[0], np.shape(output_data)[0])
        quit()

    #Threshold for an output to be considered correct
    correct_threshold = 0.5

    #Testing sums
    correct = 0
    false = 0
    confidence = 0

    for i in range(data_length):

        x_n = input_data[i]
        y_n = feed_forward(w,x_n)
        t_n = output_data[i]
        
        c = (1-y_n[ohe.one_hot_encoding_inverse(t_n)])**2

        if c < correct_threshold**2:
            correct += 1
        else:
            false += 1
        
        confidence += np.sqrt(c)
    
    confidence = 1 - confidence/testing_data_size
    
    return correct/(correct+false)

def data_set_test(w, input_data, output_data): #Returns total error from data set
    
    #Input check
    data_length = np.shape(input_data)[0]
    if np.shape(output_data)[0] != data_length:
        print("Input data shape does not equal output data shape")
        print(np.shape(input_data)[0], np.shape(output_data)[0])
        quit()

    #Summing errors
    error_sum = 0
    for i in range(data_length):

        x_val_n = input_data[i]
        t_val_n = output_data[i]
        y_val_n = feed_forward(w,x_val_n)

        error_sum += error_function_L2(w,y_val_n,t_val_n)

    return error_sum

def error_function(y_n,t_n):
    return -(t_n*np.log(y_n) + (1.0001-t_n)*np.log(1.0001-y_n))

def error_function_L2(w,y_n,t_n): #Error function with L2 regularization
    return error_function(y_n, t_n) + np.sum(np.square(w))

def feed_forward(w,x_n):
    return g(np.dot(w,x_n))

def gradient_function(x_n,y_n,t_n): #Returns gradient with given testing data
    
    return np.outer((t_n-y_n), x_n)

def gradient_function_L2(w,x_n,y_n,t_n):
    return gradient_function(x_n,y_n,t_n) + 2 * L2_coeffisient * w

def minimizing_direction(w,x,t, i, e):

    gradient = np.zeros([10,785])

    number_of_training_sets = min(batch_size,training_data_size-i)
    if number_of_training_sets <= 0: return 0

    for k in range(number_of_training_sets):

        #Finding training data
        x_n = x[i]
        y_n = feed_forward(w,x_n)
        t_n = t[i]

        #Performing gradient descent
        gradient += gradient_function_L2(w,x_n,y_n,t_n)


        i += 1
    
    gradient = gradient/number_of_training_sets

    return (-learning_rate(i, e)*gradient, i)

def gradient_descent(w, training_data_input, training_data_output):

    for e in range(epoch):
        
        i = 0
        i_last_update = 0
        update_error_vectors(w)

        #Running training data
        while i < training_data_size:
            min_dir, i = minimizing_direction(w,training_data_input,training_data_output, i, e)
            w = w - min_dir

            #Updating error vectors
            if (i - i_last_update >= training_data_size/4):
                update_error_vectors(w)
                i_last_update = i
            
        print("Epoch: ", e+1)
        
    update_error_vectors(w)

    return w
    


#Training a network
weights = gradient_descent(weights, training_data_input,training_data_output)

#plt.plot(error_vector_training, label = 'training error')
#plt.plot(error_vector_validation, label = 'validation error')
#plt.plot(error_vector_test, label = 'testing error')
plt.plot(percent_correct_training_vector, label = 'percentage correct training data')
plt.plot(percent_correct_validation_vector, label = 'percentage correct validation data')
plt.plot(percent_correct_testing_vector, label = 'percentage correct testing data')
plt.plot()
plt.legend()
plt.grid(linestyle='-', linewidth=1)
plt.show()