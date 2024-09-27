import numpy as np

# Create two matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Matrix addition
addition = matrix1 + matrix2
print("Addition:\n", addition)

# Matrix subtraction
subtraction = matrix1 - matrix2
print("Subtraction:\n", subtraction)

# Matrix multiplication (element-wise)
multiplication = matrix1 * matrix2
print("Element-wise Multiplication:\n", multiplication)

# Matrix dot product
dot_product = np.dot(matrix1, matrix2)
print("Dot Product:\n", dot_product)

# Matrix transpose
transpose = np.transpose(matrix1)
print("Transpose:\n", transpose)

# Determinant of a matrix
determinant = np.linalg.det(matrix1)
print("Determinant of matrix1:\n", determinant)
