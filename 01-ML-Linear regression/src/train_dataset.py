"""
Train a linear regression model on the California Housing dataset
using gradient descent (from scratch, NumPy only).

This script demonstrates:
- data loading
- feature scaling (standardization)
- gradient descent training
- numerical stability best practices
"""

import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Path where the dataset is expected to be stored
DATA_PATH = Path("data/california_housing.csv")


# -------------------------------------------------------------------
# Step 1: Load the dataset
# -------------------------------------------------------------------

def load_data(path: Path):
    """
    Load the California Housing dataset from a CSV file.

    Parameters
    ----------
    path : Path
        Path to the dataset CSV file.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Please run: python src/download_dataset.py"
        )

    df = pd.read_csv(path)

    # Separate features and target
    X = df.drop(columns=["price"]).values.astype(np.float64)
    y = df["price"].values.astype(np.float64)

    return X, y


# -------------------------------------------------------------------
# Step 2: Feature scaling (VERY IMPORTANT)
# -------------------------------------------------------------------

def standardize(X):
    """
    Standardize features to have:
    - mean = 0
    - standard deviation = 1

    This is required to avoid numerical instability when using
    gradient descent on real-world datasets.

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix.

    Returns
    -------
    X_scaled : np.ndarray
        Standardized features.
    mean : np.ndarray
        Feature-wise mean.
    std : np.ndarray
        Feature-wise standard deviation.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8  # small epsilon to avoid division by zero

    X_scaled = (X - mean) / std
    return X_scaled, mean, std


# -------------------------------------------------------------------
# Step 3: Train linear regression using gradient descent
# -------------------------------------------------------------------

def train(X, y, lr=0.01, epochs=3000):
    """
    Train a linear regression model using gradient descent.

    Model:
        y_hat = X @ w + b

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    y : np.ndarray
        Target values.
    lr : float
        Learning rate.
    epochs : int
        Number of training iterations.

    Returns
    -------
    w : np.ndarray
        Learned weight vector.
    b : float
        Learned bias term.
    """
    n_samples, n_features = X.shape

    # Initialize parameters
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        # Forward pass: predictions
        y_hat = X @ w + b

        # Compute error
        error = y_hat - y

        # Compute gradients (Mean Squared Error)
        dw = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * error.sum()

        # Gradient descent update
        w -= lr * dw
        b -= lr * db

    return w, b


# -------------------------------------------------------------------
# Step 4: Run the full pipeline
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Load data
    X, y = load_data(DATA_PATH)

    # Scale features
    X_scaled, mean, std = standardize(X)

    # Train model
    w, b = train(X_scaled, y)

    # Display results
    print("✅ Linear Regression trained successfully")
    print(f"Bias (b): {b:.4f}")
    print("Weights (on standardized features):")
    for i, weight in enumerate(w):
        print(f"  w{i}: {weight:.4f}")