import numpy as np

# Outputs from the 5 decision trees (continuous values indicating exam scores)
tree_outputs = [2.5, 3.0, 2.7, 2.9, 2.8]

# Random forest regression final prediction (average of tree outputs)
final_prediction = np.mean(tree_outputs)

print(f"Final exam score prediction by the random forest: {final_prediction}")
print("This is regression since we are printing a value not a classification pass/fail")