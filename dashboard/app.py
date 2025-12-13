import sys
import os

# Ensure project root is on sys.path so `src` package is importable
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    # Debug: print path added
    print(f"[DEBUG] Added project root to sys.path: {root_dir}")
# Debug: ensure 'src' directory exists and is importable
debug_src = os.path.join(root_dir, 'src')
if not os.path.isdir(debug_src):
    print(f"[DEBUG] Warning: 'src' folder not found at {debug_src}")
else:
    print(f"[DEBUG] 'src' folder exists: {debug_src}")

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from src.preprocess import load_and_preprocess


def load_model(model_path):
    obj = joblib.load(model_path)
    model = obj.get("model", obj)
    scaler = obj.get("scaler")
    return model, scaler


def main():
    st.title("Threat Detection Dashboard")

    st.sidebar.header("Configuration")
    data_file = st.sidebar.text_input("Dataset CSV", "data/dataset.csv")
    model_file = st.sidebar.text_input("Model (joblib)", "models/model.joblib")

    if not os.path.exists(data_file):
        st.warning(f"Dataset not found: {data_file}. Generate dataset using CLI first.")
    if not os.path.exists(model_file):
        st.warning(f"Model not found: {model_file}. Train model first.")

    if os.path.exists(data_file) and os.path.exists(model_file):
        df = pd.read_csv(data_file)
        st.subheader("Dataset Preview")
        st.write(df.head())

        # class distribution
        st.subheader("Class Distribution")
        counts = df['label'].value_counts().reset_index()
        counts.columns = ['label', 'count']
        counts['label'] = counts['label'].astype(str)
        fig = px.bar(counts, x='label', y='count', title='Label counts')
        st.plotly_chart(fig, width='stretch')

        model, scaler = load_model(model_file)

        # Use full dataset for metrics but split for scaling consistency
        X_train, X_test, y_train, y_test, _ = load_and_preprocess(data_file)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        st.subheader("Classification Report")
        st.text(classification_report(y_test, preds, digits=4))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig2, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig2)

        if proba is not None:
            st.subheader("ROC & Precision-Recall")
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            roc_auc = auc(fpr, tpr)
            prc_prec, prc_recall, _ = precision_recall_curve(y_test, proba[:, 1])

            fig3 = px.line(x=fpr, y=tpr, labels={'x': 'FPR', 'y': 'TPR'}, title=f'ROC Curve (AUC={roc_auc:.3f})')
            st.plotly_chart(fig3, width='stretch')

            fig4 = px.line(x=prc_recall, y=prc_prec, labels={'x': 'Recall', 'y': 'Precision'}, title='Precision-Recall Curve')
            st.plotly_chart(fig4, width='stretch')

            # Threshold adjustment
            st.subheader("Custom Threshold (to trade-off precision/recall)")
            thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
            new_preds = (proba[:, 1] >= thr).astype(int)
            st.text(classification_report(y_test, new_preds, digits=4))

        # Feature importance if tree-based
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importances")
            feat_names = df.columns[df.columns != 'label'].tolist()
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
            feat_df = feat_df.sort_values('importance', ascending=False)
            fig5 = px.bar(feat_df.head(20), x='feature', y='importance', title='Top 20 feature importances')
            st.plotly_chart(fig5, width='stretch')


if __name__ == '__main__':
    main()
