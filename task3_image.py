import pickle
import matplotlib.pyplot as plt
import numpy as np
import mnist

with open('/home/fenics/Documents/datasyn/TDT4265_A1/task3_data/weights_01', 'rb') as fp: weights = pickle.load(fp)

X_train, Y_train, X_test, Y_test = mnist.load()

num_classifications = 10
Epoch = 5

for i in range(num_classifications):
    #Average appearence of each digit
    current_number = X_train[Y_train == i]
    avg = np.mean(current_number,axis=0)
    avg_matrix = np.reshape(avg,(28,28))
    plt.imshow(avg_matrix)
    plt.show()

    #Appearance of each digit according to weigths
    weight_matrix = np.reshape(weights[Epoch][i][0:-1],(28,28))
    plt.imshow(weight_matrix)
    plt.show()
