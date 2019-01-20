import numpy as np

def one_hot_encoding(a):
    res = []
    for i in range(np.shape(a)[0]):
       r = [0,0,0,0,0,0,0,0,0,0]
       r[a[i]] = 1
       res.append(r)
    return np.array(res)

def one_hot_encoding_inverse(a):
   return int(np.dot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],a))