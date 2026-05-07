"""
Spam Detection Metrics Demonstration

This script demonstrates how to compute common classification metrics
for anomaly detection (e.g., spam detection in emails).

Dataset Summary:
----------------
Total Emails: 1000

Confusion Matrix:
                Predicted Spam    Predicted Not Spam
Actual Spam         TP = 250          FN = 50
Actual Not Spam     FP = 100         TN = 600

Metrics Computed:
-----------------
- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- Specificity
"""

# -------------------------------
# Step 1: Define confusion matrix
# -------------------------------

TP = 250  # True Positives (correctly predicted spam)
FN = 50   # False Negatives (spam predicted as not spam)
FP = 100  # False Positives (not spam predicted as spam)
TN = 600  # True Negatives (correctly predicted not spam)

# Total number of emails
total = TP + TN + FP + FN

# -------------------------------
# Step 2: Compute metrics
# -------------------------------

# Accuracy: Overall correctness
accuracy = (TP + TN) / total

# Precision: How many predicted spam emails were actually spam
precision = TP / (TP + FP)

# Recall (Sensitivity): How many actual spam emails were detected
recall = TP / (TP + FN)

# F1 Score: Harmonic mean of precision and recall
f1_score = 2 * (precision * recall) / (precision + recall)

# Specificity: How well the model identifies non-spam emails
specificity = TN / (TN + FP)

# -------------------------------
# Step 3: Display results
# -------------------------------

print("=== Spam Detection Metrics ===\n")

print(f"Total Emails: {total}\n")

print("Confusion Matrix:")
print(f"True Positives (TP): {TP}")
print(f"False Negatives (FN): {FN}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}\n")

print("Evaluation Metrics:")
print(f"Accuracy:     {accuracy:.2%}")
print(f"Precision:    {precision:.2%}")
print(f"Recall:       {recall:.2%}")
print(f"F1 Score:     {f1_score:.2%}")
print(f"Specificity:  {specificity:.2%}")

# -------------------------------
# Step 4: Optional Interpretation
# -------------------------------

print("\nInterpretation:")
print("- Accuracy shows overall correctness of the model.")
print("- Precision indicates how many flagged spam emails are truly spam.")
print("- Recall shows how many actual spam emails were detected.")
print("- F1 Score balances precision and recall.")
print("- Specificity measures how well non-spam emails are identified.")