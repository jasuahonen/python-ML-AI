import numpy as np

matrix1 = np.array([[1, 1], [1.5, 2]])
matrix2 = np.array([[20, -16], [-16, 13]])
matrix3 = np.array([[2], [-3]])

transpose = np.transpose(matrix1)

matrix4 = matrix2 * transpose * matrix3

print(matrix4)


