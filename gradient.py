import numpy as np

def gradient(f, x, input_size, a, b):
    h = 1e-10
    grad = np.zeros((input_size, 1))
    for i in range(input_size):
        delta = np.zeros((input_size, 1))
        delta[i][0] = h
        grad[i][0] = (f(x + delta, a, b) - f(x, a, b)) / h
    return grad