import mnist
import numpy as np




X_train, Y_train, X_test, Y_test = mnist.load()


w = np.zeros([10,785])

