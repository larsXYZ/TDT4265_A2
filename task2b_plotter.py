import pickle
import matplotlib.pyplot as plt
import numpy as np

#Loading stored data
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/validation_data_lambda_0_01', 'rb') as fp: validation_data_lambda_0_01 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/weight_storage_lambda_0_01', 'rb') as fp: weight_storage_lambda_0_01 = pickle.load(fp)

with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/validation_data_lambda_0_001', 'rb') as fp: validation_data_lambda_0_001 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/weight_storage_lambda_0_001', 'rb') as fp: weight_storage_lambda_0_001 = pickle.load(fp)

with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/validation_data_lambda_0_0001', 'rb') as fp: validation_data_lambda_0_0001 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/weight_storage_lambda_0_0001', 'rb') as fp: weight_storage_lambda_0_0001 = pickle.load(fp)

epoch_count = 10
readings_per_epoch = 10
x = np.linspace(0,epoch_count,10*10)

plt.plot(x[0:np.shape(validation_data_lambda_0_01)[0]],validation_data_lambda_0_01, label="lambda = 0.01")
plt.plot(x[0:np.shape(validation_data_lambda_0_001)[0]],validation_data_lambda_0_001, label="lambda = 0.001")
plt.plot(x[0:np.shape(validation_data_lambda_0_0001)[0]],validation_data_lambda_0_0001, label="lambda = 0.0001")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.show()
