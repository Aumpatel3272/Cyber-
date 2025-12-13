import os
import tempfile
from src.data import generate_synthetic
from src.train import train_model


def test_train_creates_model():
    with tempfile.TemporaryDirectory() as td:
        data_path = os.path.join(td, "dataset.csv")
        model_path = os.path.join(td, "model.joblib")
        generate_synthetic(data_path, n_samples=200, n_features=10, imbalance=0.05)
        out = train_model(data_path, out_path=model_path, n_estimators=10)
        assert os.path.exists(out)
