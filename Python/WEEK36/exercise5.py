import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Data
x1 = np.array([3.1, 3.9, 5.2, 6.9]).reshape(-1, 1)  # 1st independent variable
y = np.array([10.2, 11.5, 13.9, 15])               # Dependent variable

# a) Fit a linear regression model using x1
model = LinearRegression()
model.fit(x1, y)

# Get the slope (coefficient) and intercept of the line
slope = model.coef_[0]
intercept = model.intercept_
# print(f"Linear Regression Coefficient (Slope): {slope}")
# print(f"Intercept: {intercept}")

# b) Visualize the data points and the regression line
plt.scatter(x1, y, color='blue', label='Data Points')  # Plotting the data points

# Plotting the regression line
# plt.plot(x1, model.predict(x1), color='red', label='Regression Line')
# plt.xlabel('x1 - 1st Independent Variable')
# plt.ylabel('y - Dependent Variable')
# plt.title('Linear Regression Fit')
# plt.legend()
# plt.show()

# c) Calculate the correlation between x1 and y
correlation = np.corrcoef(x1.flatten(), y)[0, 1]
#print(f"Correlation between x1 and y: {correlation}")

# d) Calculate the coefficient of determination (R-squared)
r_squared = r2_score(y, model.predict(x1))
#print(f"Coefficient of Determination (R-squared): {r_squared}")

#Additional calculations

#Residuals
predictions = model.predict(x1)
residuals = y - predictions
print(f"Residuals: {residuals}")

#Mean absolute error
mae = mean_absolute_error(y, predictions)
print(f"MAE: {mae}")

#Mean squared error
mse = mean_squared_error(y, predictions)
print(f"MSE: {mse}")

#Root mean squared error
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")