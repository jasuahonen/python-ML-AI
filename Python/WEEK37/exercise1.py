import numpy as np
import matplotlib.pyplot as plt

# Plot the sigmoid-function of one variable with
# a) zero bias
# b) bias of 3
# c) bias of -3 and variable coefficient 2
# Plot them in the same figure. What do you observe? How do you see the bias in the graphs? How about variable coefficient?


# Sigmoid function
def sigmoid(x, a=1, b=0):
    return 1 / (1 + np.exp(-(a * x + b)))

# Range of x values
x = np.linspace(-10, 10, 400)

# Compute the sigmoid functions with different biases and coefficients
y_zero_bias = sigmoid(x)             # a = 1, b = 0
y_bias_3 = sigmoid(x, b=3)           # a = 1, b = 3
y_bias_minus3_coef2 = sigmoid(x, a=2, b=-3)  # a = 2, b = -3

# Plotting the functions
plt.figure(figsize=(8, 6))
plt.plot(x, y_zero_bias, label='Sigmoid (a=1, b=0)', color='blue')
plt.plot(x, y_bias_3, label='Sigmoid (a=1, b=3)', color='green')
plt.plot(x, y_bias_minus3_coef2, label='Sigmoid (a=2, b=-3)', color='red')

# Add title and labels
plt.title('Sigmoid Function with Different Biases and Coefficient')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.legend()
plt.grid(True)
plt.show()


## The blue is the normal sigmoid function b = 0,
## green is b = 3 which moves the curve horizontally to the left
## red is a = 2, which steepens the curve since the coefficient grows
## red also has b = -3 which moves the curve horizontally to the right

