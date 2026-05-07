# Python script to demonstrate the various evaluation metrics for anomaly detection

# Import the required modules
import numpy as np

# Define the ground truth labels and predicted labels
ground_truth = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
predicted_labels = np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 0])

# Calculate the number of true positives (TP), false positives (FP),
# true negatives (TN), and false negatives (FN)

TP = np.sum((ground_truth == 1) & (predicted_labels == 1))
FP = np.sum((ground_truth == 0) & (predicted_labels == 1))
TN = np.sum((ground_truth == 0) & (predicted_labels == 0))
FN = np.sum((ground_truth == 1) & (predicted_labels == 0))

# Calculate the True Positive Rate (TPR), False Positive Rate
# (FPR), True Negative Rate (TNR), and False Negative Rate (FNR)
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
TNR = TN / (TN + FP)
FNR = FN / (FN + TP)

# Calculate the Precision, Recall, and F1 Score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the results
print(f"True Positive Rate (TPR): {TPR:.2f}")
print(f"False Positive Rate (FPR): {FPR:.2f}")
print(f"True Negative Rate (TNR): {TNR:.2f}")
print(f"False Negative Rate (FNR): {FNR:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")