# Python script to build a customer churn prediction model using Gaussian Naive Bayes (GNE)
# The target variables is (1 = churned, 0 = retained)
# Features used are customer data such as
# monthly usage (minutes / data)
# Average call duration
# Customer tenure (months)
# Billing type (re-paid or post-paid)

# Import the required modules
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


# Function to generate customer data
def generate_customer_data(n_samples=500):
    """
    Generate synthetic customer churn dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of customer records to generate (default is 500).

    Returns
    -------
    DataFrame
        A pandas DataFrame containing customer features and churn labels.
    """
    np.random.seed(42)  # For reproducibility

    # Features
    monthly_usage = np.random.normal(loc=300, scale=50, size=n_samples)  # minutes / data usage
    avg_call_duration = np.random.normal(loc=5, scale=2, size=n_samples)  # minutes
    tenure = np.random.randint(1, 60, size=n_samples)  # months
    billing_type = np.random.choice(["pre-paid", "post-paid"], size=n_samples)  # categorrical

    # Churn probability influenced by tenure & billing type
    churn_prob = (0.3 * (monthly_usage < 250) + (0.4 * (tenure < 12)) + (0.2 * (billing_type == "pre-paid")))
    churn = np.random.binomial(1, np.clip(churn_prob, 0, 1))  # Target variable

    # Create a dataframe
    data = pd.DataFrame({
        "monthly_usage": monthly_usage,
        "avg_call_duration": avg_call_duration,
        "tenure": tenure,
        "billing_type": billing_type,
        "churn": churn
    })

    return data


# Function to preprocess the data
def preprocess_data(data):
    """
    Preprocess dataset for modeling:
    - Encode categorical variables
    - Split into train/test sets

    Parameters
    ----------
    data : DataFrame
        Customer churn dataset.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test arrays ready for modeling.
    """
    # Encode billing type
    encoder = LabelEncoder()
    data['billing_type'] = encoder.fit_transform(data['billing_type'])

    # Features and target
    X = data[["monthly_usage", "avg_call_duration", "tenure", "billing_type"]]
    y = data["churn"]

    # Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Function to train & measure / evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train Gaussian Naive Bayes model and evaluate performance.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : arrays
        Training and testing datasets.

    Returns
    -------
    None
    """
    # Initialize the model
    model = GaussianNB()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Classification report: {classification_report(y_test, y_pred)}")
    print(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}")


# Run the application
if __name__ == "__main__":
    # Step 1. Generate synthetic data
    data = generate_customer_data()
    print(f"Sample data: {data.head(8)}")

    # Step 2, Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Step 3. Train and evaluate
    train_and_evaluate(X_train, y_train, X_test, y_test)
