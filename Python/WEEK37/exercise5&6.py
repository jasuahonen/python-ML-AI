import numpy as np
from sklearn.linear_model import LogisticRegression

#exercise 5

# Data
age = np.array([36, 42, 46, 50, 54, 61, 66, 70]).reshape(-1, 1)
smoking = np.array([0, 0, 10, 5, 15, 5, 5, 0]).reshape(-1, 1)
heart_disease = np.array([0, 0, 1, 0, 1, 0, 1, 1])

# Independent variables: concatenate age and smoking to form a feature matrix
X = np.concatenate((age, smoking), axis=1)

# Dependent variable: heart disease
y = heart_disease

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Print model coefficients (age, smoking) and intercept (bias term)
print("Coefficients (age, smoking):", model.coef_)
print("Intercept (bias):", model.intercept_)

# Predict heart disease probabilities for the same data
probabilities = model.predict_proba(X)[:, 1]
print("Predicted probabilities:", probabilities)

# Predict the binary outcome for heart disease
predictions = model.predict(X)
print("Predicted binary outcomes:", predictions)

print("")


#exercise 6


# Data for new cases
new_cases = np.array([
    [73, 0],   # a) Person of age 73, who does not smoke
    [73, 7],   # b) Person of age 73, who smokes 7 cigarettes per day
    [36, 17]   # c) Person of age 36, who smokes 17 cigarettes per day
])

# Predict the logistic probability for the new cases
probabilities_new = model.predict_proba(new_cases)[:, 1]

# Print the probabilities for each case
print("Predicted probabilities for new cases:")
print(f"a) Age 73, no smoking: {probabilities_new[0]}")
print(f"b) Age 73, smokes 7 cigarettes/day: {probabilities_new[1]}")
print(f"c) Age 36, smokes 17 cigarettes/day: {probabilities_new[2]}")
