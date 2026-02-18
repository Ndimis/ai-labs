import numpy as np

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
def make_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, size=(n, 1))
    y = 3 * X[:, 0] + 2 + rng.normal(0, 0.5, size=n)
    return X, y

def train(X, y, lr=0.01, epochs=1500):
    w, b = 0.0, 0.0
    n = len(y)
    for _ in range(epochs):
        y_hat = w * X[:, 0] + b
        dw = (2/n) * ((y_hat - y) * X[:, 0]).sum()
        db = (2/n) * (y_hat - y).sum()
        w -= lr * dw
        b -= lr * db
    return w, b

if __name__ == '__main__':
    X, y = make_data()
    w, b = train(X, y)
    print(f"Learned parameters: w={w:.2f}, b={b:.2f}")