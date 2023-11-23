import numpy as np

def projected_gradient(c):
    inv = np.linalg.inv(c @ c.T)
    prj = np.eye(50) - (c.T @ inv @ c)
    return prj