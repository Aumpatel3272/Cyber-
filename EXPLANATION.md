# Threat Detection with ML — Deep Dive

## 1. Data Generation (`src/data.py`)

### What It Does
Generates a **synthetic dataset** that mimics real network traffic with a small percentage of threats.

### The Logic

```python
X, y = make_classification(
    n_samples=5000,           # 5000 network traffic records
    n_features=20,            # 20 network features (packet size, timing, protocol, etc.)
    n_informative=12,         # 12 features actually predict threats
    n_redundant=2,            # 2 redundant features (correlated with others)
    weights=[0.98, 0.02],     # 98% normal, 2% threats (realistic imbalance)
    flip_y=0.01               # 1% label noise (real data is messy)
)
```

### Why This Approach?
- **Class Imbalance:** Real threats are rare (~2%), so the model must learn from limited positive examples
- **Feature Noise:** 10% of features are uninformative (adds realism)
- **Label Noise:** 1% mislabeled data (happens in reality when labeling is automated)

### Example Features (Hypothetical)
- `f0`: Packet size (bytes)
- `f1`: Inter-arrival time (seconds)
- `f2`: Protocol type (TCP/UDP/ICMP)
- `f3`: Source IP entropy
- `f4`: Destination port commonality
- ... (16 more features)

### What We Could Add
```python
# 1. Real network feature engineering
def extract_network_features(pcap_file):
    """Extract features from actual network traffic (PCAP file)."""
    # Packet size distribution, flow duration, port scanning patterns, etc.
    pass

# 2. Dataset imbalance handling with SMOTE
def generate_with_smote(n_samples=5000):
    """Use SMOTE to generate synthetic minority samples."""
    from imblearn.over_sampling import SMOTE
    X, y = make_classification(...)
    smote = SMOTE()
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced
```

---

## 2. Preprocessing (`src/preprocess.py`)

### What It Does
Prepares data for training by splitting, scaling, and handling missing values.

### The Logic

```
Raw Data (5000 samples × 20 features)
    ↓
Train/Test Split (80%/20%, stratified)
    ↓
Feature Scaling (StandardScaler)
    ↓
Normalized Data Ready for Training
```

### Why Standardization Matters

**Without scaling:**
- Feature 1 range: [0, 10000] (packet size)
- Feature 2 range: [0, 1] (IP entropy)
- The algorithm treats feature 1 as more important (larger numbers) = **bias**

**After StandardScaler (mean=0, std=1):**
- Feature 1: [-2.5, 2.5] (normalized)
- Feature 2: [-2.5, 2.5] (normalized)
- All features contribute equally = **fair comparison**

**Formula:**
```
x_scaled = (x - mean) / std_dev
```

### Stratified Split (Why It Matters)

**Bad approach (random split):**
```
Train: 3900 normal, 50 threats (1.3%)
Test:  1100 normal, 50 threats (4.3%)  ← Different distribution!
```

**Good approach (stratified split):**
```
Train: 3920 normal, 80 threats (2.0%)
Test:  980 normal, 20 threats (2.0%)   ← Same distribution ✓
```

### What We Could Add
```python
# 1. Handle missing values
def preprocess_with_imputation(X, strategy='mean'):
    """Fill missing values with mean/median/forward-fill."""
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X)
    return X_imputed

# 2. Feature selection (drop irrelevant features)
def select_best_features(X_train, y_train, n_features=15):
    """Keep only top N features that best predict threats."""
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=n_features)
    X_selected = selector.fit_transform(X_train, y_train)
    return X_selected, selector

# 3. Detect outliers
def remove_outliers(X, y, threshold=3):
    """Remove extreme outliers (anomalies in training data)."""
    from scipy import stats
    z_scores = np.abs(stats.zscore(X))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], y[mask]
```

---

## 3. RandomForest Training (`src/train.py`)

### What It Does
Trains 100 decision trees that vote on whether traffic is a threat.

### The Logic

**Decision Tree (Single Tree):**
```
                Is Packet Size > 1000?
               /                        \
             Yes                         No
            /                              \
    Is Entropy > 0.8?                  Probably Normal
   /                      \
 Yes → Likely Threat    No → Probably Normal
```

**RandomForest (100 Trees):**
```
Input: New network packet
    ↓
Pass through Tree 1 → Prediction: THREAT
Pass through Tree 2 → Prediction: NORMAL
Pass through Tree 3 → Prediction: THREAT
... (97 more trees)
    ↓
Vote: 60 trees say THREAT, 40 say NORMAL
    ↓
Final Decision: THREAT (majority vote)
```

### Why RandomForest?
- **Handles Imbalance:** Naturally good at detecting minority class (threats)
- **No Scaling Needed:** Uses splits, not distances (unlike KNN or SVM)
- **Robust:** Averaging reduces overfitting
- **Fast Inference:** Predictions in milliseconds

### The Trade-off

**Hyperparameters:**
```python
n_estimators=100    # More trees = better but slower
max_depth=None      # Unlimited depth = overfitting risk
min_samples_split=2 # Minimum samples per node
```

**Better tuning:**
```python
# Balanced approach
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,           # Limit depth to prevent overfitting
    min_samples_split=10,   # Require 10 samples per node
    class_weight='balanced' # Give more weight to minority class
)
```

### What We Could Add
```python
# 1. Hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    """Find best parameters using GridSearchCV."""
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier()
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    return grid.best_estimator_

# 2. Get feature importance
def feature_importance(model, feature_names):
    """Show which features are most important for predictions."""
    importances = model.feature_importances_
    for name, importance in sorted(
        zip(feature_names, importances), 
        key=lambda x: x[1], 
        reverse=True
    ):
        print(f"{name}: {importance:.4f}")

# 3. Handle class imbalance
def train_with_class_weight(X_train, y_train):
    """Give more penalty for misclassifying threats."""
    clf = RandomForestClassifier(
        class_weight='balanced',  # Automatically adjusts for imbalance
        n_estimators=200
    )
    clf.fit(X_train, y_train)
    return clf
```

---

## 4. Evaluation Metrics (`src/evaluate.py`)

### Understanding the Output

```
              precision    recall  f1-score   support

           0     0.9780    1.0000    0.9889       976
           1     1.0000    0.0833    0.1538        24
```

### What Each Metric Means

**For Normal Traffic (Class 0):**
- **Precision 0.9780:** Of 1000 times we said "NORMAL", 978 were actually normal
  - Formula: `True Negatives / (True Negatives + False Positives)`
  - Low false alarms ✓

- **Recall 1.0000:** We correctly identified 100% of normal traffic
  - Formula: `True Negatives / (True Negatives + False Negatives)`
  - We missed 0 normal packets ✓

**For Threats (Class 1):**
- **Precision 1.0000:** When we said "THREAT", we were always right (0 false alarms)
  - Great: No innocent traffic blocked

- **Recall 0.0833:** We only caught 8.3% of actual threats (2 out of 24)
  - Bad: We miss 91.7% of attacks
  - Reason: Very few threats in training data, model needs rebalancing

**F1-Score (Harmonic Mean):**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Balances precision and recall
- For threats: F1 = 0.1538 (poor, because recall is very low)

### Confusion Matrix
```
                Predicted
           NORMAL  THREAT
Actual  NORMAL   976      0     ← Great! 0 false alarms
        THREAT    22      2     ← Bad! Missed 22 threats
```

### What We Could Add
```python
# 1. ROC Curve (visualize trade-off)
def plot_roc_curve(y_test, y_pred_proba):
    """Plot Receiver Operating Characteristic curve."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# 2. Threshold optimization
def find_optimal_threshold(y_test, y_pred_proba):
    """Find threshold that balances precision and recall."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
    # Find threshold where F1 is maximized
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

# 3. Cross-validation
def cross_validate(model, X, y, cv=5):
    """Test on 5 different splits for robustness."""
    from sklearn.model_selection import cross_validate
    scores = cross_validate(
        model, X, y, 
        cv=cv,
        scoring=['precision', 'recall', 'f1']
    )
    return scores
```

---

## 5. The ML Concept — Why This Approach for Threat Detection?

### The Problem
- **High Volume:** 1 billion packets/day on a network
- **Rare Threats:** 0.1% are actual attacks (99.9% normal)
- **Speed Critical:** Must classify in milliseconds
- **Evolving Threats:** New attack patterns constantly emerge

### The Solution: Supervised Learning

**Step 1: Historical Data**
```
Normal packets (labeled NORMAL)
Attack packets (labeled THREAT)
        ↓
Train a model to recognize patterns
        ↓
Apply to new packets in real-time
```

**Why RandomForest over alternatives?**

| Approach | Pros | Cons |
|----------|------|------|
| **Rule-Based** | Interpretable, fast | Can't catch new attacks, brittle |
| **Anomaly Detection** | Unsupervised, flexible | High false alarm rate, hard to tune |
| **Deep Learning (NN)** | Very accurate | Slow, needs massive data, black box |
| **RandomForest** ✓ | Fast, balanced, robust, no scaling | Medium accuracy, less flexible |

### Real-World Improvements

```python
# What We Could Add for Production

# 1. Online Learning (retrain incrementally)
def update_model_online(model, X_new, y_new):
    """Retrain on newly labeled threats (streaming updates)."""
    model.fit(X_new, y_new)  # Partial fit available in some algorithms
    return model

# 2. Ensemble with multiple models
def ensemble_models(X_test):
    """Combine RandomForest + XGBoost + Neural Network."""
    pred_rf = rf.predict_proba(X_test)
    pred_xgb = xgb.predict_proba(X_test)
    pred_nn = nn.predict_proba(X_test)
    # Average predictions
    ensemble_pred = (pred_rf + pred_xgb + pred_nn) / 3
    return ensemble_pred

# 3. Anomaly scoring (confidence)
def get_threat_confidence(model, X):
    """How confident is the model in threat predictions?"""
    predictions = model.predict_proba(X)
    threat_confidence = predictions[:, 1]  # Probability of threat class
    return threat_confidence

# 4. Logging and monitoring
def log_predictions(timestamp, packet_id, prediction, confidence):
    """Track all predictions for audit and improvement."""
    log_entry = {
        'timestamp': timestamp,
        'packet_id': packet_id,
        'prediction': 'THREAT' if prediction == 1 else 'NORMAL',
        'confidence': confidence,
        'model_version': '1.0'
    }
    # Save to database or file
    pass
```

### Real Example: Why Our Results Matter

**In Production (1M packets/day):**
- Model predicts 2000 threats
- **Precision 1.0:** All 2000 are real threats (0 innocent packets blocked)
- **Recall 0.083:** We catch 8.3% = 166 threats, miss 1834

**Impact:**
- ✓ Zero false alarms (no legitimate user impact)
- ✗ Misses most attacks (security risk)

**To improve:** Use `class_weight='balanced'` or collect more threat data

---

## Summary: The Full Pipeline

```
GENERATE DATA
    ↓
├─ 5000 synthetic samples
└─ 20 features, 2% threat rate

PREPROCESS
    ↓
├─ Split: 3920 train (normal), 80 train (threat)
├─ Scale: StandardScaler
└─ Result: Test set (1000 samples)

TRAIN
    ↓
└─ 100 RandomForest trees learn threat patterns

EVALUATE
    ↓
└─ Precision/Recall/F1-Score report
    ├─ Normal: High accuracy
    └─ Threats: High precision, low recall

DEPLOY
    ↓
└─ Real-time threat scoring for new packets
```
