import numpy as np

def f(x, a, b):
    abs_values = np.zeros(500)
    R = np.matmul(a.T, x.flatten()) + b.flatten()
    for i in range(500):
        abs_values[i] = np.abs(R[i])
    return np.max(abs_values)