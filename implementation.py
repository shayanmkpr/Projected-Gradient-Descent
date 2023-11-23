import numpy as np
import matplotlib.pyplot as plt
from function_def import f
from constant_step import constant_step
from constant_dist import constant_dist
from first_order import first_order
from second_order import second_order


x0 = 10 * np.ones((50, 1))
a = np.array(np.random.normal(0 , 4 , (50 , 500)))
c = np.array(np.random.normal(0, 4, (50,1)))
b = np.random.normal(0 , 4 , (1 , 500))
d = np.random.normal(0, 4, 1)


minimizer_1, values_1, x_history_1 = constant_step(f, x0, 100, 1e-10, a , b , c , d)
minimizer_2, values_2, x_history_2 = constant_dist(f, x0, 100, 1e-10, a , b , c , d)
minimizer_3, values_3, x_history_3 = first_order(f, x0, 100, 1e-10, a , b , c , d)
minimizer_4, values_4, x_history_4 = second_order(f, x0, 100, 1e-10, a , b , c , d)

plt.plot(range(len(values_1)), values_1, label='Constant Step Size')
plt.plot(range(len(values_2)), values_2, label='Constant Distance')
plt.plot(range(len(values_3)), values_3, label='1/k')
plt.plot(range(len(values_4)), values_4, label='1/sqrt(k)')
plt.xlabel('Iteration (k)')
plt.ylabel('Function Value')
plt.legend()
plt.title('Convergence of Function Value')

plt.show()