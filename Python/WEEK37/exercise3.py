import numpy as np

#exercise a)

bias_a = 5
variable_a = 2
coefficient_a = 0.5

logit_a = coefficient_a * variable_a + bias_a
probability_a = 1 / (1+np.exp(-logit_a))

print("probability a:", probability_a)

#exercise b)

bias_b = 3
variable_b = 4
coefficient_b = -1.5

logit_b = coefficient_b * variable_b + bias_b
probability_b = 1 / (1+np.exp(-logit_b))

print("probability b:", probability_b)

#exercise c)

bias_c = 5
variables_c = np.array([2,3])
coefficients_c = np.array([0.5,-4])

logit_c = np.dot(coefficients_c, variables_c) + bias_c
probability_c = 1 / (1+np.exp(-logit_c))

print("probability c:", probability_c)