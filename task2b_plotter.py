import pickle
import matplotlib.pyplot as plt
import numpy as np

epoch = 10

#Loading stored data
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/validation_data_lambda_0_01', 'rb') as fp: validation_data_lambda_0_01 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/weight_storage_lambda_0_01', 'rb') as fp: weight_storage_lambda_0_01 = pickle.load(fp)

with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/validation_data_lambda_0_001', 'rb') as fp: validation_data_lambda_0_001 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/weight_storage_lambda_0_001', 'rb') as fp: weight_storage_lambda_0_001 = pickle.load(fp)

with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/validation_data_lambda_0_0001', 'rb') as fp: validation_data_lambda_0_0001 = pickle.load(fp)
with open('/home/l/Documents/TDT4265/Assignment1/task2b_data/weight_storage_lambda_0_0001', 'rb') as fp: weight_storage_lambda_0_0001 = pickle.load(fp)

plt.plot(validation_data_lambda_0_01, label="lambda = 0.01")
plt.plot(validation_data_lambda_0_001, label="lambda = 0.001")
plt.plot(validation_data_lambda_0_0001, label="lambda = 0.0001")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.show()