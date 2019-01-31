import pickle
import matplotlib.pyplot as plt
import numpy as np

#Loading stored data
with open('/home/l/Documents/TDT4265/Assignment1/task4_data/loss_vector_validation_backprop', 'rb') as fp: loss_vector_validation_backprop = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task4_data/loss_vector_validation_nesterov', 'rb') as fp: loss_vector_validation_nesterov = pickle.load(fp)

epoch_count = 10
number_of_error_check_per_epoch = 10
x = np.linspace(0,epoch_count,epoch_count*number_of_error_check_per_epoch+2)


#Plot validation result
plt.plot(x, loss_vector_validation_backprop, label="Normal backpropagation")
plt.plot(x, loss_vector_validation_nesterov, label="With Nesterov Momentum")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss function validation data set")
plt.grid(color='grey', linestyle='-', linewidth=1)
plt.show()
