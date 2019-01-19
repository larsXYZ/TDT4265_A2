import mnist
import numpy as np
import one_hot_encoding as ohe
import matplotlib.pyplot as plt

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

print(np.shape(X_train), np.shape(Y_train))
print(np.shape(X_test), np.shape(Y_test))
input()

#Hypervariables
epoch = 20
batch_size = 100
initial_learning_rate = 0.01 #Annealing learning rate
T = 300000

def learning_rate(i, e): #Annealing learning rate
    return initial_learning_rate/(1+(i+training_data_size*e)/T)

def g(y_n):
    return 1/(1+np.exp(-y_n))

def test(w,testing_data_input,testing_data_output):
    
    correct_threshold = 0.7
    
    correct = 0
    false = 0

    confidence = 0

    for i in range(testing_data_size):

        x_n = testing_data_input[i]
        y_n = feed_forward(w,x_n)
        t_n = testing_data_output[i]
        
        c = (t_n-y_n)**2

        if c < correct_threshold**2:
            correct += 1
        else:
            false += 1
        
        confidence += np.sqrt(c)
    
    confidence = 1 - confidence/testing_data_size
    
    print("Testing size: ", testing_data_size, " | " , correct, " correct | ", false, " false | percentage: ", correct/(correct+false), " | Avg confidence: ", confidence)
        


def validation_test(w, x_val, t_val): #Returns total error from validation data
    
    error_validation = 0
    for i in range(validation_data_size):
        x_val_n = x_val[i]
        t_val_n = t_val[i]
        y_val_n = feed_forward(w,x_val_n)

        error_validation += error_function(y_val_n,t_val_n)
    return error_validation

def error_function(y_n,t_n):
    return -(t_n*np.log(y_n) + (1.0001-t_n)*np.log(1.0001-y_n))

def feed_forward(w,x_n):
    return g(np.dot(w,x_n))

def gradient_function(x_n,y_n,t_n): #Returns gradient with given testing data
    
    return np.outer((t_n-y_n), x_n)

def minimizing_direction(w,x,t, i, e):

    gradient = np.zeros([1,785])
    error_training = 0

    number_of_training_sets = min(batch_size,training_data_size-i)
    if number_of_training_sets <= 0: return 0

    
    for k in range(number_of_training_sets):

        #Finding training data
        x_n = x[i]
        y_n = feed_forward(w,x_n)
        t_n = t[i]

        #Performing gradient descent
        error_training += error_function(y_n,t_n)
        gradient += gradient_function(x_n,y_n,t_n)

        i += 1
    
    gradient = gradient/number_of_training_sets
    error_training

    return (-learning_rate(i, e)*gradient,error_training, i)

def gradient_descent(training_data_input, training_data_output):
    
    error_vector_training = []
    error_vector_validation = []
    w = np.random.rand(1,785)

    for e in range(epoch):
        i = 0

        error_training_sum = 0

        while i < training_data_size:

            min_dir, error_training, i = minimizing_direction(w,training_data_input,training_data_output, i, e)
            w = w - min_dir
            error_training_sum += error_training

        #Checking validation data
        error_validation = validation_test(w,validation_data_input,validation_data_output)

        error_vector_training.append(error_training_sum/training_data_size)
        error_vector_validation.append(error_validation/validation_data_size)


        print("Epoch:", e+1, " | Current training-error: ", error_training_sum/training_data_size, " | Current validation-error: ", error_validation/validation_data_size, " | Current learning rate: ", learning_rate(i, e))
        
    
    return (w, error_vector_training, error_vector_validation)
    


#Training a network
weights, error_vector_training, error_vector_validation = gradient_descent(training_data_input,training_data_output)
test(weights,testing_data_input,testing_data_output)

plt.plot(error_vector_training)
plt.plot(error_vector_validation)
plt.show()