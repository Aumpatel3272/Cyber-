from sklearn.datasets import make_classification
import pandas as pd
import os


def generate_synthetic(out_path, n_samples=5000, n_features=20, imbalance=0.02, random_state=42):
    """Generate a synthetic dataset with a small minority class to simulate threats.

    Saves a CSV with features named f0..f{n-1} and a `label` column (0=normal,1=threat).
    """
    weights = [1.0 - imbalance, imbalance]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.1),
        n_clusters_per_class=1,
        weights=weights,
        flip_y=0.01,
        random_state=random_state,
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    # quick local test
    generate_synthetic("data/dataset.csv")
