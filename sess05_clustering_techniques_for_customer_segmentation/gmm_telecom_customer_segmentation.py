# Python script to demonstrate Gaussian Mixture Model (GMM) for customer segmentation

# --------------------------------------------------------------------------------------
# Step 0. Import the required modules
# --------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

np.random.seed(42)  # For reproducibility


# --------------------------------------------------------------------------------------
# Step 1. Load data
# --------------------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fix TotalCharges (common issue in the telco dataset)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


# --------------------------------------------------------------------------------------
# Step 2. Preprocessing
# --------------------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Select features
    features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    # Ensure numeric + handle missing values
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
        df[feature] = df[feature].fillna(df[feature].median())

    X = df[features].values

    # FIX: Fit before transform
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df


# --------------------------------------------------------------------------------------
# Step 3. PCA Visualization
# --------------------------------------------------------------------------------------
def plot_pca(X_scaled, df):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 5))
    plt.scatter(
        X_2d[:, 0],   # FIXED
        X_2d[:, 1],   # FIXED
        c=df["Churn"],
        cmap="coolwarm",
        alpha=0.6
    )

    plt.title("PCA Projection (colored by Churn)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Churn (0 = No, 1 = Yes)")
    plt.show()

    return X_2d


# --------------------------------------------------------------------------------------
# Step 4. Find the optimal number of components
# --------------------------------------------------------------------------------------
def find_optimal_components(X_scaled, max_k=8):
    ks = range(1, max_k + 1)
    bics = []

    for k in ks:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_scaled)
        bics.append(gmm.bic(X_scaled))

    plt.figure(figsize=(6, 4))
    plt.plot(ks, bics, marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("BIC")
    plt.title("Selecting Number of Clusters (BIC)")
    plt.show()

    best_k = ks[np.argmin(bics)]
    print(f"\nOptimal number of clusters (BIC): {best_k}")

    return best_k  # FIXED


# --------------------------------------------------------------------------------------
# Step 5. Train GMM Model
# --------------------------------------------------------------------------------------
def train_gmm(X_scaled, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    return gmm, labels, probs


# --------------------------------------------------------------------------------------
# Step 6. Visualize Clusters
# --------------------------------------------------------------------------------------
def plot_clusters(X_2d, labels, probs):
    confidence = probs.max(axis=1)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        X_2d[:, 0],   # FIXED
        X_2d[:, 1],   # FIXED
        c=labels,
        cmap="tab10",
        s=confidence * 60 + 10,
        alpha=0.7
    )

    plt.title("GMM Clusters (Point Size = Confidence)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter)
    plt.show()  # FIXED


# --------------------------------------------------------------------------------------
# Step 7. Segment Interpretation
# --------------------------------------------------------------------------------------
def describe_segments(df, labels):
    df = df.copy()
    df["Segment"] = labels

    summary = df.groupby("Segment")[[
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]].mean()

    print("\nSegment summary (Mean values)")
    print(summary.round(2))

    print("\nBusiness interpretation")

    for seg in summary.index:
        row = summary.loc[seg]
        description = []

        if row["tenure"] < 20:
            description.append("New Customers")
        else:
            description.append("Long-term Customers")

        if row["MonthlyCharges"] > 70:
            description.append("High Spend")
        else:
            description.append("Low/Moderate Spend")

        if row["Churn"] > 0.4:
            description.append("High churn risk")
        else:
            description.append("Low churn risk")

        print(f"\nSegment {seg}: {', '.join(description)}")


# --------------------------------------------------------------------------------------
# Step 8. Main function
# --------------------------------------------------------------------------------------
def main():
    print("Loading data...")

    data_file = (
        Path(__file__).resolve().parent.parent
        / "files"
        / "kaggle_blastchar_telco_customer_churn.csv"
    )

    try:
        df = load_data(str(data_file))
    except FileNotFoundError:
        print("File not found")
        return
    except PermissionError:
        print(f"Permission denied: {data_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"{data_file} is empty")
        return
    except pd.errors.ParserError:
        print(f"{data_file} could not be parsed")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    print("Preprocessing data...")
    X_scaled, df = preprocess(df)

    print("PCA Visualization...")
    X_2d = plot_pca(X_scaled, df)

    print("Selecting number of clusters...")
    k = find_optimal_components(X_scaled)

    print("Training GMM...")
    gmm, labels, probs = train_gmm(X_scaled, k)

    print("Visualizing clusters...")
    plot_clusters(X_2d, labels, probs)

    print("Interpreting segments...")
    describe_segments(df, labels)


# Run the application
if __name__ == "__main__":
    main()