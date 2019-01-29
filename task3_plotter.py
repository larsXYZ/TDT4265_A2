import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/error_vector_test', 'rb') as fp: error_vector_test = pickle.load(fp)
with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/error_vector_training', 'rb') as fp: error_vector_training = pickle.load(fp)
with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/error_vector_validation', 'rb') as fp: error_vector_validation = pickle.load(fp)
with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/percent_correct_testing_vector', 'rb') as fp: percent_correct_testing_vector = pickle.load(fp)
with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/percent_correct_training_vector', 'rb') as fp: percent_correct_training_vector = pickle.load(fp)
with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/percent_correct_validation_vector', 'rb') as fp: percent_correct_validation_vector = pickle.load(fp)

plt.plot(error_vector_test, label='Testing')
plt.plot(error_vector_training, label='Training')
plt.plot(error_vector_validation, label = 'Validation')
plt.ylabel('Loss function value')
plt.legend()
plt.show()


for i in range(len(percent_correct_testing_vector)):
    percent_correct_training_vector[i] = percent_correct_training_vector[i]*100
    percent_correct_validation_vector[i] = percent_correct_validation_vector[i]*100
    percent_correct_testing_vector[i] = percent_correct_testing_vector[i]*100

plt.plot(percent_correct_testing_vector, label='Testing')
plt.plot(percent_correct_training_vector, label='Training')
plt.plot(percent_correct_validation_vector, label='Validation')
plt.ylabel('Percentage correct')
plt.legend()
plt.show()
