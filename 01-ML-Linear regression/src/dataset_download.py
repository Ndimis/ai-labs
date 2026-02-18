from sklearn.datasets import fetch_california_housing
import pandas as pd
from pathlib import Path

def main():
    print("Downloading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)

    df = data.frame
    df = df.rename(columns={"MedHouseVal": "price"})

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    out_file = out_dir / "california_housing.csv"
    df.to_csv(out_file, index=False)

    print(f"Dataset saved to {out_file}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()