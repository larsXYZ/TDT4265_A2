import mnist
import numpy as np
import one_hot_encoding as ohe
import matplotlib.pyplot as plt
import pickle

#Data settings
training_data_size = 11000
validation_data_size = 500
testing_data_size = 1000

#Loading data from MNIST dataset
X_train, Y_train, X_test, Y_test = mnist.load()

#Reducing values to between 0 - 1
X_train = X_train/255
X_test = X_test/255

#Performing the "bias trick"
X_train = np.concatenate((X_train,np.ones([60000,1])), axis=1)
X_test = np.concatenate((X_test,np.ones([10000,1])), axis=1)

#Filtering out all other values than 2 and 3
X_train = X_train[(Y_train == 2) | (Y_train == 3)]
Y_train = (Y_train[(Y_train == 2) | (Y_train == 3)] == 2).astype(int)

X_test = X_test[(Y_test == 2) | (Y_test == 3)]
Y_test = (Y_test[(Y_test == 2) | (Y_test == 3)] == 2).astype(int)

#Selecting training data and validation data
training_data_input = X_train[0:training_data_size,:].copy()
training_data_output = Y_train[0:training_data_size].copy()

validation_data_input = X_train[training_data_size:training_data_size+validation_data_size].copy()
validation_data_output = Y_train[training_data_size:training_data_size+validation_data_size].copy()

testing_data_input = X_test[-testing_data_size:].copy()
testing_data_output = Y_test[-testing_data_size:].copy()

print("Training and testing samples: ", np.shape(Y_train)[0], ", ", np.shape(Y_test)[0] )
print("Training data shape:",np.shape(training_data_input), np.shape(training_data_output))
print("Validation data shape:",np.shape(validation_data_input), np.shape(validation_data_output))
print("Testing data shape:",np.shape(testing_data_input), np.shape(testing_data_output))
input()

#Hypervariables
epoch = 15
batch_size = 5
initial_learning_rate = 0.001 #Initial annealing learning rate
L2_coeffisient = 0.0001 #L2 coefficient punishing model complexity, bigger -> less complex weights
T = 300000 #Annealing learning rate time constant, bigger -> slower decrease of learning rate
early_stopping_threshold = 3 # If validation score increases several times in a row, we cancel

#Error storage
number_of_error_check_per_epoch = 10
error_check_interval = training_data_size / number_of_error_check_per_epoch
error_vector_training = []
error_vector_validation = []
error_vector_test = []
percent_correct_training_vector = []
percent_correct_validation_vector = []
percent_correct_testing_vector = []

#Initializing weights 
weights = np.random.rand(1,785)

#Keeping track of our performance testing
global training_sets_evaluated
global last_performance_check
training_sets_evaluated = 0
last_performance_check = 0

def performance_test(w):
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

def learning_rate(): #Annealing learning rate

    return initial_learning_rate/(1+(training_sets_evaluated)/T)

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
        
        c = (t_n-y_n)**2

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
    return -(t_n*np.log(0.001+y_n) + (1-t_n)*np.log(1.001-y_n))

def error_function_L2(w,y_n,t_n): #Error function with L2 regularization
    return error_function(y_n, t_n) + L2_coeffisient*np.sum(np.square(w))

def feed_forward(w,x_n):
    return g(np.dot(w,x_n))

def gradient_function(x_n,y_n,t_n): #Returns gradient with given testing data
    
    return np.outer((t_n-y_n), x_n)

def gradient_function_L2(w,x_n,y_n,t_n):
    return gradient_function(x_n,y_n,t_n) + 2 * L2_coeffisient * w

def minimizing_direction(w,x,t, i, e):

    gradient = np.zeros([1,785])

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

        #Checking if we should do a performance check of our network
        global training_sets_evaluated
        global last_performance_check
        training_sets_evaluated += 1
        if training_sets_evaluated - last_performance_check >= error_check_interval:
            performance_test(w)            
            last_performance_check = training_sets_evaluated
    
    gradient = gradient/number_of_training_sets

    return (-learning_rate()*gradient, i)

def gradient_descent(w, training_data_input, training_data_output):

    #Storing previous weights for early stopping
    weight_storage = []
    validation_data_early_stopping = []
    weight_storage.append(w)

    #Recording network performance for plotting
    performance_test(w)

    #Recording network performance for early stopping
    validation_data_early_stopping.append(error_vector_validation[-1][0])

    for e in range(epoch):
        
        i = 0

        #Running training data
        while i < training_data_size:
            min_dir, i = minimizing_direction(w,training_data_input,training_data_output, i, e)
            w = w - min_dir
            
        #Recording state for early stopping
        weight_storage.append(w)
        validation_data_early_stopping.append(error_vector_validation[-1][0])

        print("Epoch: ", e+1, " | Learning rate: ", learning_rate(), " | Validation error: ", validation_data_early_stopping[-1], " | Percentage correctly guessed (validation data set)",  percent_correct_validation_vector[-1])

        #Early stopping test
        validation_only_increasing = True
        if (e >= early_stopping_threshold-1):
            for i in range(early_stopping_threshold):
                if validation_data_early_stopping[-i-1] <= validation_data_early_stopping[-i-2]:
                    validation_only_increasing = False
                    break
            if validation_only_increasing:
                print("EARLY STOPPING: Validation error function increased ", early_stopping_threshold, " times in a row. Stopping training")
                return weight_storage[-1-i], weight_storage
    
    #Recording network performance
    performance_test(w)

    #Finding best weight from weight history
    best_weight = validation_data_early_stopping[-1]
    best_epoch = epoch
    for i in range(early_stopping_threshold):
        if validation_data_early_stopping[-1-i] < validation_data_early_stopping[best_epoch]:
            best_weight = weight_storage[-1-i]
            best_epoch = epoch - i
            
    print("Training finished: Returning weights from epoch ", best_epoch, "With validation score ", validation_data_early_stopping[best_epoch])
    return best_weight, weight_storage
    
#Training a network
weights, weight_storage = gradient_descent(weights, training_data_input,training_data_output)

#Serializing list to file, in order to plot for several lambda values in task 2.2 bcd
with open ('validation_data_lambda_0_0001', 'wb') as fp: pickle.dump(percent_correct_validation_vector, fp)
with open ('weight_storage_lambda_0_0001', 'wb') as fp: pickle.dump(weight_storage, fp)