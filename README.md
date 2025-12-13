Threat Detection with ML
========================

Minimal end-to-end scaffold for threat/anomaly detection using a supervised ML baseline.

Quick start

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Generate synthetic dataset, train, and evaluate:

```bash
python -m src.main generate --out data/dataset.csv
python -m src.main train --data data/dataset.csv --out models/model.joblib
python -m src.main evaluate --data data/dataset.csv --model models/model.joblib
```

Files of interest:
- `src/data.py` (data generation)
- `src/preprocess.py` (simple preprocessing)
- `src/train.py` (training baseline)
- `src/evaluate.py` (evaluation)

Artifacts
---------

Use the `scripts/collect_artifacts.py` helper to gather the generated dataset and trained model into the `artifacts/` folder for submission:

```powershell
python scripts/collect_artifacts.py --out artifacts
```

If you already ran the example commands, `artifacts/` will contain `dataset.csv` and `model.joblib` after running the script.

Dashboard
---------

A Streamlit dashboard is provided in `dashboard/app.py` to visualize the dataset, classification report, confusion matrix, ROC curve, and feature importances.

Run it locally (make sure you have dependencies installed):

```powershell
streamlit run dashboard/app.py
```

The sidebar accepts paths for `data` and `model` (defaults to `data/dataset.csv` and `models/model.joblib`).
# Threat-Detection
