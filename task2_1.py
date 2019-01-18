import mnist
import numpy as np
import one_hot_encoding as ohe
import matplotlib.pyplot as plt



#Loading data
training_data_size = 5000
validation_data_size = 100
testing_data_size = 1000

X_train, Y_train, X_test, Y_test = mnist.load()
X_train = X_train/255
X_test = X_test/255
X_train = np.concatenate((X_train,np.ones([60000,1])), axis=1)
X_test = np.concatenate((X_test,np.ones([10000,1])), axis=1)

testing_data = X_test[-testing_data_size:,:].copy()

training_data_output = ohe.one_hot_encoding(Y_train[0:training_data_size])
training_data_input = X_train[0:training_data_size,:].copy()

validation_data_output = ohe.one_hot_encoding(Y_train[training_data_size:training_data_size+validation_data_size])
validation_data_input = X_train[training_data_size:training_data_size+validation_data_size].copy()



#Hypervariables
epoch = 20
batch_size = 50
initial_learning_rate = 0.001 #Annealing learning rate
T = 30

def learning_rate(i, e): #Annealing learning rate
    return 0.001
    #return initial_learning_rate/(1+(i+batch_size*e)/T)

def validation_test(w, x_val, t_val):
    
    error_validation = 0
    for i in range(validation_data_size):
        x_val_n = x_val[i]
        t_val_n = t_val[i]
        y_val_n = feed_forward(w,x_val_n)
        error_validation += error_function(y_val_n,t_val_n)
    error_validation /= validation_data_size
    return error_validation

def error_function(y_n,t_n):
    print(np.sum(np.dot(t_n,np.log(y_n))))
    return t_n*np.log(y_n)

def feed_forward(w,x_n):
    return np.dot(w,x_n)

def gradient_function(x_n,y_n,t_n): #Returns gradient with given testing data
    
    return np.outer((t_n-y_n), x_n)

def minimizing_direction(w,x,t, x_val, t_val, i, e):

    gradient = np.zeros([10,785])
    error_training = 0
    error_validation = 0

    number_of_training_sets = min(batch_size,training_data_size-i)
    if number_of_training_sets == 0: return 0

    for k in range(number_of_training_sets):

        #Finding training data
        x_n = x[i+k]
        y_n = feed_forward(w,x_n)
        t_n = t[i+k]

        #Performing gradient descent
        error_training += error_function(y_n,t_n)
        gradient += gradient_function(x_n,y_n,t_n)

    #Checking validation data
    error_validation = validation_test(w,x_val,t_val)
    
    gradient = gradient/number_of_training_sets
    error_training = error_training/number_of_training_sets

    return (-learning_rate(i, e)*gradient,error_training,error_validation)

def gradient_descent(training_data_input, training_data_output):
    
    error_vector_training = []
    error_vector_validation = []
    w = np.random.rand(10,785)

    for e in range(epoch):
        i = 0
        while i < training_data_size:

            min_dir, error_training, error_validation = minimizing_direction(w,training_data_input,training_data_output, validation_data_input, validation_data_output, i, e)
            w = w - min_dir
            i += batch_size
            error_vector_training.append(error_training)
            error_vector_validation.append(error_validation)

        print("Epoch:", e+1, " | Current training-error: ", error_training, " | Current validation-error: ", error_validation, " | Current learning rate: ", learning_rate(0, epoch))
        
    
    return (w, error_vector_training, error_vector_validation)
    


#Training a network
weights, error_vector_training, error_vector_validation = gradient_descent(training_data_input,training_data_output)


plt.plot(error_vector_training)
plt.show()