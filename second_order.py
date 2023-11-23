import numpy as np
from gradient import gradient
from projection import projected_gradient

def second_order(f, x0, max_iter, tol, a, b, c, d):
    x = x0.reshape(-1, 1)
    input_size = len(x)
    val = np.zeros(max_iter)
    x_history = [x.flatten()]
    print("Iteration\tdifferential Value\t\t\t\tFunction Value")

    for k in range(1, max_iter):
        step = 1/np.sqrt(k)
        x = x - step * (projected_gradient(c) @ gradient(f, x, 50, a, b))
        val[k] = f(x, a, b)
        x_history.append(x.flatten())
        print(f"{k}\t{val[k]}")
        # Check for convergence
        if np.linalg.norm(gradient(f, x, input_size, a, b)) < tol:
            break
    return x, val[:k+1], np.array(x_history)