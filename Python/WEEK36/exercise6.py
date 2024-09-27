import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data
x1 = np.array([3.1, 3.9, 5.2, 6.9])  # 1st independent variable
x2 = np.array([9, 7.5, 6, 5])        # 2nd independent variable
y = np.array([10.2, 11.5, 13.9, 15])  # Dependent variable

# Combine x1 and x2 into a single 2D array for the model
X = np.column_stack((x1, x2))

# a) Fit a linear regression model using x1 and x2
model = LinearRegression()
model.fit(X, y)

# Get the coefficients and intercept of the model
coefficients = model.coef_
intercept = model.intercept_

# print(f"Linear Regression Coefficients: {coefficients}")
# print(f"Intercept: {intercept}")

# b) Calculate the coefficient of determination (R-squared) for this model
# r_squared = r2_score(y, model.predict(X))
# print(f"Coefficient of Determination (R-squared): {r_squared}")


# Predictions
predictions = model.predict(X)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y, predictions)
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, predictions)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

#The MAE, MSE and RMSE are smaller since we are using to variables which
# basically make the model more accurate when calculating errors