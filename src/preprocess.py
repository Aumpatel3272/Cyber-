import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_and_preprocess(csv_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("Dataset must have a 'label' column")
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def save_numpy(out_path, X, y):
    import os
    os.makedirs(out_path, exist_ok=True)
    np.savez_compressed(f"{out_path}/split.npz", X=X, y=y)
