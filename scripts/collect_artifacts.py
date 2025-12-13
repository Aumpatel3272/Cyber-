"""Collect artifacts (dataset and model) into an `artifacts/` folder.

Usage:
    python scripts/collect_artifacts.py --out artifacts

The script copies `data/dataset.csv` and `models/model.joblib` if they exist.
"""
import argparse
import os
import shutil


def collect(data_path="data/dataset.csv", model_path="models/model.joblib", out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    copied = []
    if os.path.exists(data_path):
        shutil.copy2(data_path, os.path.join(out_dir, os.path.basename(data_path)))
        copied.append(data_path)
    if os.path.exists(model_path):
        shutil.copy2(model_path, os.path.join(out_dir, os.path.basename(model_path)))
        copied.append(model_path)
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    copied = collect(args.data, args.model, args.out)
    if copied:
        print("Copied:")
        for p in copied:
            print(" -", p)
        print("Artifacts are in:", args.out)
    else:
        print("No artifacts found. Generate and train first, then re-run this script.")


if __name__ == "__main__":
    main()
