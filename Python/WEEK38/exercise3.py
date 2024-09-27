import numpy as np

# Function to calculate entropy
def entropy(p):
    # Handle the case where p = 0 to avoid log(0)
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

# Calculate left and right entropy impurities
left_entropy = calculate_entropy_impurity(left_sick_count, left_healthy_count)
right_entropy = calculate_entropy_impurity(right_sick_count, right_healthy_count)

# Calculate total entropy (weighted average of left and right entropy impurities)
total_patients = len(S_x) + len(H_x)
total_entropy = (left_sick_count + left_healthy_count) / total_patients * left_entropy + \
                (right_sick_count + right_healthy_count) / total_patients * right_entropy

# Output results
print(f"Left-side entropy impurity: {left_entropy}")
print(f"Right-side entropy impurity: {right_entropy}")
print(f"Total entropy impurity: {total_entropy}")
