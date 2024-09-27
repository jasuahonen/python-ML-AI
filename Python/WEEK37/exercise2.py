import numpy as np

# For the heart disease model we did logistic regression fitting together, plot the sigmoid function. The parameters were approximately -5,3 and 0.111 for bias and age, respectively.
# a) At what age do we have 50% probability for heart disease according to our model
# b) At what age do we have over 90% probability for heart disease according to our model
# c) Predict the (logistic) probability for heart disease for a 73 year old person
# Analyze the model critically.

# Given parameters
coefficient_age = 0.111
bias = -5.3

# a) Solve for age when probability is 50%
age_50_percent = bias / coefficient_age

# b) Solve for age when probability is 90%
p_90 = 0.9
logit_90 = np.log(p_90 / (1 - p_90))
age_90_percent = (logit_90 - bias) / coefficient_age

# c) Predict probability for a 73-year-old
age_73 = 73
logit_73 = coefficient_age * age_73 + bias
probability_73 = 1 / (1 + np.exp(-logit_73))

print(age_50_percent, age_90_percent, probability_73)

#The age when probability is 50% = -47,74747
#This tells us there might be something skewed in the model

#90% probability age = 67,5 years
#

#Probability for a 73yo = 94,2%
