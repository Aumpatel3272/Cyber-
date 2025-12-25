from sklearn.ensemble import RandomForestClassifier
import joblib
from .preprocess import load_and_preprocess
import os


def train_model(data_csv, out_path="models/model.joblib", n_estimators=200, random_state=42):
    """Train a RandomForest with class weight balancing and save the model to `out_path`.

    Uses class_weight='balanced' to handle severe class imbalance (98% normal, 2% threats).
    Returns the output path.
    """
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess(data_csv)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',  # Handle class imbalance
        max_depth=15,             # Prevent overfitting
        min_samples_split=10,     # Require min samples per node
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler}, out_path)
    return out_path


if __name__ == "__main__":
    train_model("data/dataset.csv")
