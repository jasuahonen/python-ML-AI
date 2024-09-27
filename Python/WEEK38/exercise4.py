import numpy as np

# Function to calculate entropy
def entropy(p):
    if p == 0:
        return 0
    return -p * np.log2(p)

# Function to calculate entropy impurity for a group
def calculate_entropy_impurity(sick_count, healthy_count):
    total = sick_count + healthy_count
    if total == 0:
        return 0
    p_sick = sick_count / total
    p_healthy = healthy_count / total
    return entropy(p_sick) + entropy(p_healthy)

# Function to calculate Gini impurity for a group
def calculate_gini_impurity(sick_count, healthy_count):
    total = sick_count + healthy_count
    if total == 0:
        return 0
    p_sick = sick_count / total
    p_healthy = healthy_count / total
    return 1 - (p_sick ** 2 + p_healthy ** 2)

# Data for Sick (S) and Healthy (H) patients
S_x = np.array([0.2, 0.35, 0.6, 0.7, 0.85, 1.35])
H_x = np.array([1.1, 1.2, 1.4, 1.45, 1.6, 1.65, 1.65])

# Splitting criteria (example: threshold = 1.0)
threshold = 1.0

# Left and Right groups based on the threshold
S_left = S_x[S_x <= threshold]
S_right = S_x[S_x > threshold]
H_left = H_x[H_x <= threshold]
H_right = H_x[H_x > threshold]

# Count the number of sick and healthy patients in left and right groups
left_sick_count = len(S_left)
right_sick_count = len(S_right)
left_healthy_count = len(H_left)
right_healthy_count = len(H_right)

# Calculate the number of sick and healthy patients in the parent node (before the split)
parent_sick_count = len(S_x)
parent_healthy_count = len(H_x)

# Calculate Gini and Entropy for the parent node (before split)
parent_gini = calculate_gini_impurity(parent_sick_count, parent_healthy_count)
parent_entropy = calculate_entropy_impurity(parent_sick_count, parent_healthy_count)

# Calculate left and right Gini impurities
left_gini = calculate_gini_impurity(left_sick_count, left_healthy_count)
right_gini = calculate_gini_impurity(right_sick_count, right_healthy_count)

# Calculate left and right Entropy impurities
left_entropy = calculate_entropy_impurity(left_sick_count, left_healthy_count)
right_entropy = calculate_entropy_impurity(right_sick_count, right_healthy_count)

# Calculate the total number of patients (for weighting purposes)
total_patients = len(S_x) + len(H_x)

# Weighted average Gini impurity of children (left and right)
weighted_gini = (left_sick_count + left_healthy_count) / total_patients * left_gini + \
                (right_sick_count + right_healthy_count) / total_patients * right_gini

# Weighted average Entropy impurity of children (left and right)
weighted_entropy = (left_sick_count + left_healthy_count) / total_patients * left_entropy + \
                   (right_sick_count + right_healthy_count) / total_patients * right_entropy

# Calculate Information Gain for Gini
information_gain_gini = parent_gini - weighted_gini

# Calculate Information Gain for Entropy
information_gain_entropy = parent_entropy - weighted_entropy

# Output results
print(f"Parent Gini Impurity: {parent_gini}")
print(f"Left Gini Impurity: {left_gini}")
print(f"Right Gini Impurity: {right_gini}")
print(f"Information Gain (Gini): {information_gain_gini}")
print(f"Parent Entropy Impurity: {parent_entropy}")
print(f"Left Entropy Impurity: {left_entropy}")
print(f"Right Entropy Impurity: {right_entropy}")
print(f"Information Gain (Entropy): {information_gain_entropy}")
