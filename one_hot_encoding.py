import numpy as np

def one_hot_encoding(a):
    res = []
    for i in range(np.shape(a)[0]):
       r = [0,0,0,0,0,0,0,0,0,0]
       r[a[i]] = 1
       res.append(r)
    return np.array(res)