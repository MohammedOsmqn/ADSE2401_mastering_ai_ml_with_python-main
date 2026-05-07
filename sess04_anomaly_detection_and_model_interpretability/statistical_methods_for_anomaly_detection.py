import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# -----------------------------
# Step 1: Data
# -----------------------------
np.random.seed(42)

normal_data = np.random.normal(loc=50, scale=5, size=200)
anomalies = np.array([90, 95, 100, 10, 5])

data = np.concatenate((normal_data, anomalies))
data = np.asarray(data).ravel()  # IMPORTANT FIX


# -----------------------------
# Step 2: Z-Score
# -----------------------------
def z_score_detection(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold


# -----------------------------
# Step 3: Modified Z-Score
# -----------------------------
def modified_z_score_detection(data, threshold=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    if mad == 0:
        return np.zeros_like(data, dtype=bool)

    modified_z = 0.6745 * (data - median) / mad
    return np.abs(modified_z) > threshold


# -----------------------------
# Step 4: IQR (FIXED)
# -----------------------------
def iqr_detection(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    iqr = Q3 - Q1

    lower_bound = Q1 - 1.5 * iqr
    upper_bound = Q3 + 1.5 * iqr

    return (data < lower_bound) | (data > upper_bound)


# -----------------------------
# Step 5: Gaussian
# -----------------------------
def gaussian_detection(data, threshold=0.01):
    mean = np.mean(data)
    std = np.std(data)

    probs = stats.norm.pdf(data, mean, std)
    return probs < threshold


# -----------------------------
# Step 6: Apply methods
# -----------------------------
z_anomalies = z_score_detection(data)
mod_z_anomalies = modified_z_score_detection(data)
iqr_anomalies = iqr_detection(data)
gaussian_anomalies = gaussian_detection(data)


# -----------------------------
# Step 7: Plot
# -----------------------------
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

methods = [
    ("Z-Score", z_anomalies),
    ("Modified Z-Score", mod_z_anomalies),
    ("IQR", iqr_anomalies),
    ("Gaussian", gaussian_anomalies),
]

for n, (title, mask) in enumerate(methods, 1):
    plt.subplot(2, 2, n)

    mask = np.asarray(mask).astype(bool)

    anomaly_idx = np.where(mask)[0]

    plt.scatter(range(len(data)), data, label="Normal", alpha=0.6)

    plt.scatter(
        anomaly_idx,
        data[anomaly_idx],
        color="red",
        label="Anomaly",
        s=80,
    )

    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()

plt.tight_layout()
plt.show()