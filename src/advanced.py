"""Advanced functions for threat detection model enhancement.

These are optional improvements that can be integrated into the main pipeline.
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import stats
import matplotlib.pyplot as plt


# ============================================================================
# 1. DATA ENHANCEMENT FUNCTIONS
# ============================================================================

def remove_outliers(X, y, threshold=3):
    """Remove extreme outliers using z-score method.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels
        threshold: Z-score threshold (default 3 = 99.7% of data)
    
    Returns:
        X_clean, y_clean: Data without outliers
    """
    z_scores = np.abs(stats.zscore(X))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], y[mask]


def preprocess_with_imputation(X, strategy='mean'):
    """Fill missing values using specified strategy.
    
    Args:
        X: Feature matrix with possible NaNs
        strategy: 'mean', 'median', 'most_frequent', or 'constant'
    
    Returns:
        X_imputed: Data without missing values
    """
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X)
    return X_imputed, imputer


def select_best_features(X_train, y_train, n_features=15):
    """Select top N features that best predict threats.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_features: Number of features to keep
    
    Returns:
        X_selected: Reduced feature set
        selector: Fitted selector (for transforming test data)
    """
    selector = SelectKBest(f_classif, k=n_features)
    X_selected = selector.fit_transform(X_train, y_train)
    return X_selected, selector


# ============================================================================
# 2. MODEL TRAINING ENHANCEMENTS
# ============================================================================

def tune_hyperparameters(X_train, y_train, cv=5):
    """Find best RandomForest hyperparameters using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
    
    Returns:
        best_model: Trained model with best parameters
        results: Grid search results
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best F1 score: {grid.best_score_:.4f}")
    
    return grid.best_estimator_, grid


def train_with_class_weight(X_train, y_train, n_estimators=200):
    """Train RandomForest with automatic class imbalance handling.
    
    Uses class_weight='balanced' to penalize misclassification of minority class.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
    
    Returns:
        model: Trained classifier
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf


def get_feature_importance(model, feature_names, top_n=10):
    """Extract and display feature importance.
    
    Args:
        model: Trained RandomForest model
        feature_names: List of feature names
        top_n: Show top N features
    
    Returns:
        importance_dict: Dictionary of features and their importance scores
    """
    importances = model.feature_importances_
    importance_dict = dict(zip(feature_names, importances))
    
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} Important Features:")
    print("-" * 40)
    for feature, importance in sorted_features[:top_n]:
        print(f"{feature}: {importance:.4f}")
    
    return importance_dict


# ============================================================================
# 3. EVALUATION ENHANCEMENTS
# ============================================================================

def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation with multiple metrics.
    
    Args:
        model: Classifier to validate
        X: Features
        y: Labels
        cv: Number of folds
    
    Returns:
        scores_dict: Dictionary with mean/std for each metric
    """
    scoring = {
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc'
    }
    
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    print("\nCross-Validation Results (5-fold):")
    print("-" * 50)
    for metric in ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']:
        test_scores = scores[f'test_{metric}']
        print(f"{metric.upper()}:")
        print(f"  Mean: {test_scores.mean():.4f}")
        print(f"  Std:  {test_scores.std():.4f}")
    
    return scores


def plot_roc_curve(y_test, y_pred_proba, title="ROC Curve"):
    """Plot Receiver Operating Characteristic curve.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    
    return roc_auc


def plot_precision_recall(y_test, y_pred_proba, title="Precision-Recall Curve"):
    """Plot Precision-Recall curve (better for imbalanced datasets).
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def find_optimal_threshold(y_test, y_pred_proba):
    """Find optimal classification threshold that balances precision/recall.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        optimal_threshold: Best threshold value
        best_f1: Best F1-score achieved
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"Best F1-Score: {best_f1:.4f}")
    
    return optimal_threshold, best_f1


def get_threat_confidence(model, X):
    """Get confidence score for threat predictions.
    
    Args:
        model: Trained classifier
        X: Feature matrix
    
    Returns:
        threat_confidence: Probability of threat for each sample
    """
    predictions = model.predict_proba(X)
    threat_confidence = predictions[:, 1]  # Probability of class 1 (threat)
    return threat_confidence


# ============================================================================
# 4. PRODUCTION UTILITIES
# ============================================================================

def predict_with_confidence(model, X, threshold=0.5):
    """Make predictions with confidence scores.
    
    Args:
        model: Trained classifier
        X: Feature matrix
        threshold: Decision threshold (default 0.5)
    
    Returns:
        predictions: Binary predictions
        confidence: Confidence scores
    """
    proba = model.predict_proba(X)
    threat_confidence = proba[:, 1]
    predictions = (threat_confidence >= threshold).astype(int)
    return predictions, threat_confidence


def log_prediction(packet_id, prediction, confidence, timestamp=None):
    """Log a single prediction (useful for audit trails).
    
    Args:
        packet_id: Identifier for the packet
        prediction: 0 (normal) or 1 (threat)
        confidence: Confidence score
        timestamp: When prediction was made
    """
    log_entry = {
        'packet_id': packet_id,
        'prediction': 'THREAT' if prediction == 1 else 'NORMAL',
        'confidence': float(confidence),
        'timestamp': timestamp or str(np.datetime64('now'))
    }
    # In production, save to database or log file
    print(f"[LOG] {log_entry}")
    return log_entry


if __name__ == "__main__":
    print("Advanced threat detection functions loaded.")
    print("Import these functions into your training pipeline to enhance the model.")
