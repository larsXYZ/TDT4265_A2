import pickle
import matplotlib.pyplot as plt
import numpy as np

#Loading stored data
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/validation_data_lambda_0_01', 'rb') as fp: validation_data_lambda_0_01 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/weight_storage_lambda_0_01', 'rb') as fp: weight_storage_lambda_0_01 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/length_of_weight_vector_0_01', 'rb') as fp: length_of_weight_vector_0_01 = pickle.load(fp)

with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/validation_data_lambda_0_001', 'rb') as fp: validation_data_lambda_0_001 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/weight_storage_lambda_0_001', 'rb') as fp: weight_storage_lambda_0_001 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/length_of_weight_vector_0_001', 'rb') as fp: length_of_weight_vector_0_001 = pickle.load(fp)

with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/validation_data_lambda_0_0001', 'rb') as fp: validation_data_lambda_0_0001 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/weight_storage_lambda_0_0001', 'rb') as fp: weight_storage_lambda_0_0001 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/length_of_weight_vector_0_0001', 'rb') as fp: length_of_weight_vector_0_0001 = pickle.load(fp)

epoch_count = 50
number_of_error_check_per_epoch = 10
x = np.linspace(0,epoch_count,epoch_count*number_of_error_check_per_epoch)

print(weight_storage_lambda_0_01[-1])
print(np.shape(validation_data_lambda_0_01))

#Plot validation result
plt.plot(x[1:np.shape(validation_data_lambda_0_01)[0]-1],length_of_weight_vector_0_01[1:-1], label="lambda = 0.01")
plt.plot(x[1:np.shape(validation_data_lambda_0_001)[0]-1],length_of_weight_vector_0_001[1:-1], label="lambda = 0.001")
plt.plot(x[1:np.shape(validation_data_lambda_0_0001)[0]-1],length_of_weight_vector_0_0001[1:-1], label="lambda = 0.0001")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Weight vector length (L2 norm) ")
plt.grid(color='grey', linestyle='-', linewidth=1)
#plt.imshow(np.reshape(weight_storage_lambda_0_001[-1][0][:-1],(28,28)))
plt.show()
