import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/error_vector_test', 'rb') as fp: error_vector_test = pickle.load(fp)
with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/error_vector_training', 'rb') as fp: error_vector_training = pickle.load(fp)
with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/error_vector_validation', 'rb') as fp: error_vector_validation = pickle.load(fp)
with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/percent_correct_testing_vector', 'rb') as fp: percent_correct_testing_vector = pickle.load(fp)
with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/percent_correct_training_vector', 'rb') as fp: percent_correct_training_vector = pickle.load(fp)
with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/percent_correct_validation_vector', 'rb') as fp: percent_correct_validation_vector = pickle.load(fp)

#Scaling the x-axes to match epochs
epoch_count = 50
number_of_error_check_per_epoch = 10
x = np.linspace(0,epoch_count,epoch_count*number_of_error_check_per_epoch)

#Plotting error function
plt.plot(x[0:np.shape(error_vector_test)[0]],error_vector_test, label='Testing')
plt.plot(x[0:np.shape(error_vector_training)[0]],error_vector_training, label='Training')
plt.plot(x[0:np.shape(error_vector_validation)[0]],error_vector_validation, label = 'Validation')
plt.ylabel('Loss function value')
plt.xlabel('Epoch')
plt.grid(color='grey', linestyle='-', linewidth=1)
plt.legend()
plt.show()


#Converting to percentage
for i in range(len(percent_correct_testing_vector)):
    percent_correct_training_vector[i] = percent_correct_training_vector[i]*100
    percent_correct_validation_vector[i] = percent_correct_validation_vector[i]*100
    percent_correct_testing_vector[i] = percent_correct_testing_vector[i]*100

#Plotting percentage correct
plt.plot(x[2:np.shape(percent_correct_testing_vector)[0]],percent_correct_testing_vector[2:], label='Testing')
plt.plot(x[2:np.shape(percent_correct_training_vector)[0]],percent_correct_training_vector[2:], label='Training')
plt.plot(x[2:np.shape(percent_correct_validation_vector)[0]],percent_correct_validation_vector[2:], label='Validation')
plt.ylabel('Percentage correct')
plt.xlabel('Epoch')
plt.grid(color='grey', linestyle='-', linewidth=1)
plt.legend()
plt.show()
