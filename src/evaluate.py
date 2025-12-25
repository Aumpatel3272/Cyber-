import joblib
from .preprocess import load_and_preprocess
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import pandas as pd


def evaluate(data_csv, model_path):
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess(data_csv)
    obj = joblib.load(model_path)
    model = obj.get("model", obj)
    preds = model.predict(X_test)
    
    # Get probability-based confidence scores
    proba = model.predict_proba(X_test)
    confidence = proba[:, 1]  # Probability of threat class (class 1)
    
    # Classification report
    report = classification_report(y_test, preds, digits=4)
    cm = confusion_matrix(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    
    # Feature importance (if tree-based model)
    feature_importance_dict = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        n_features = len(importances)
        feature_names = [f"f{i}" for i in range(n_features)]
        feature_importance_dict = dict(zip(feature_names, importances))
    
    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confidence_scores": confidence.tolist(),
        "feature_importance": feature_importance_dict,
        "average_threat_confidence": float(confidence[y_test == 1].mean()) if sum(y_test == 1) > 0 else 0.0
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    out = evaluate(args.data, args.model)
    print(out["report"])
    print("Confusion matrix:")
    print(out["confusion_matrix"])
    print(f"\nAverage Threat Confidence: {out['average_threat_confidence']:.4f}")
    
    # Display top 10 important features
    if out["feature_importance"]:
        print("\nTop 10 Important Features:")
        sorted_features = sorted(out["feature_importance"].items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            print(f"  {feature}: {importance:.4f}")
