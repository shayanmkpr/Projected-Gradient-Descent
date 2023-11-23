import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from function_def import f

m = 500
n = 50
x_example = np.ones(n)
a_example = np.random.normal(0, 4, (n, m))
b_example = np.random.normal(0, 4, (m, 1))

# Choose one element of x to vary
element_to_vary = 1
x_values = np.linspace(-10, 10, 100)  # Generate a range of values for the chosen element
f_values = [f(x_example, a_example, b_example) for x_example[element_to_vary] in x_values]

# Plot the values
plt.plot(x_values, f_values)
plt.xlabel(f'Value of x[{element_to_vary}]')
plt.ylabel('Function Value')
plt.title('Variation of f with respect to one element of x')
plt.show()