# =============================================================
# Credit Card Fraud Detection System
# Works with any CSV — auto-detects the target column
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ==============================================================
# STEP 1: DATA COLLECTION
# ==============================================================
print("=" * 60)
print("STEP 1: Data Collection")
print("=" * 60)

CSV_FILE = "creditcard.csv"   # <-- change filename if needed
df = pd.read_csv(CSV_FILE)

print(f"Dataset shape : {df.shape}")
print(f"Columns found : {df.columns.tolist()}")

# --- Auto-detect the target (fraud label) column ---
TARGET_CANDIDATES = [
    "Class", "class", "fraud", "Fraud", "isFraud", "is_fraud",
    "label", "Label", "target", "Target", "FraudFound_P"
]
target_col = None
for col in TARGET_CANDIDATES:
    if col in df.columns:
        target_col = col
        break

# Fallback: last column if nothing matched
if target_col is None:
    target_col = df.columns[-1]
    print(f"[WARNING] Could not auto-detect target column. "
          f"Using last column: '{target_col}'")
else:
    print(f"Target column detected: '{target_col}'")

print(f"\nValue counts in '{target_col}':")
print(df[target_col].value_counts())

# Ensure binary 0/1 encoding
unique_vals = sorted(df[target_col].unique())
if set(unique_vals) != {0, 1}:
    # Map the minority class to 1 (fraud), majority to 0
    minority = df[target_col].value_counts().idxmin()
    df[target_col] = (df[target_col] == minority).astype(int)
    print(f"Re-encoded '{target_col}': '{minority}' → 1 (Fraud), rest → 0 (Legit)")

fraud_count = df[target_col].sum()
legit_count = (df[target_col] == 0).sum()
print(f"\nFraud cases      : {fraud_count} ({fraud_count/len(df)*100:.4f}%)")
print(f"Legitimate cases : {legit_count}")
print(f"\nFirst 5 rows:\n{df.head()}")


# ==============================================================
# STEP 2: DATA PREPROCESSING
# ==============================================================
print("\n" + "=" * 60)
print("STEP 2: Data Preprocessing")
print("=" * 60)

# 2a. Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Dropped {before - len(df)} duplicate rows. Remaining: {len(df)}")

# 2b. Handle missing values
missing = df.isnull().sum().sum()
if missing > 0:
    df.fillna(df.median(numeric_only=True), inplace=True)
    print(f"Filled {missing} missing values with column medians.")
else:
    print("No missing values found.")

# 2c. Feature / Target split
X = df.drop(columns=[target_col])
y = df[target_col]

# 2d. Drop any non-numeric columns
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    X.drop(columns=non_numeric, inplace=True)
    print(f"Dropped non-numeric columns: {non_numeric}")

# 2e. Scale 'Amount' and 'Time' if they exist (common in Kaggle dataset)
scaler = StandardScaler()
cols_to_scale = [c for c in ["Amount", "Time", "amount", "time"] if c in X.columns]
if cols_to_scale:
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    print(f"Scaled columns: {cols_to_scale}")
else:
    # Scale all features if no specific columns found
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print("Scaled all feature columns using StandardScaler.")

print(f"Feature matrix shape: {X.shape}")


# ==============================================================
# STEP 3: TRAIN-TEST SPLIT  (80% train / 20% test)
# ==============================================================
print("\n" + "=" * 60)
print("STEP 3: Train-Test Split (80 / 20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set  : {X_train.shape[0]} samples")
print(f"Testing set   : {X_test.shape[0]} samples")
print(f"Fraud in train: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
print(f"Fraud in test : {y_test.sum()}  ({y_test.mean()*100:.4f}%)")


# ==============================================================
# STEP 4: MODEL TRAINING
# ==============================================================
print("\n" + "=" * 60)
print("STEP 4: Model Training")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        class_weight="balanced", n_jobs=-1
    ),
}

trained = {}
for name, model in models.items():
    print(f"  Training {name}...", end=" ", flush=True)
    model.fit(X_train, y_train)
    trained[name] = model
    print("done.")


# ==============================================================
# STEP 5: MODEL EVALUATION
# ==============================================================
print("\n" + "=" * 60)
print("STEP 5: Model Evaluation")
print("=" * 60)

results = {}
for name, model in trained.items():
    y_pred = model.predict(X_test)
    results[name] = {
        "model"    : model,
        "y_pred"   : y_pred,
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1"       : f1_score(y_test, y_pred, zero_division=0),
        "cm"       : confusion_matrix(y_test, y_pred),
    }

# --- Comparison table ---
print(f"\n{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1 Score':>9}")
print("-" * 62)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f}"
          f" {r['recall']:>8.4f} {r['f1']:>9.4f}")

# --- Detailed classification reports ---
for name, r in results.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, r["y_pred"],
                                target_names=["Legitimate", "Fraud"],
                                zero_division=0))

# --- Plot: Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"])
    ax.set_title(f"{name}\nAcc: {r['accuracy']:.4f}  F1: {r['f1']:.4f}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
print("Saved: confusion_matrices.png")

# --- Plot: Metric Comparison Bar Chart ---
metric_keys   = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
model_names   = list(results.keys())
x     = np.arange(len(metric_labels))
width = 0.25
colors = ["#1565C0", "#6A1B9A", "#E65100"]

fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name][m] for m in metric_keys]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("Saved: model_comparison.png")


# ==============================================================
# STEP 6: SELECT BEST MODEL  (highest F1 Score)
# ==============================================================
print("\n" + "=" * 60)
print("STEP 6: Best Performing Model")
print("=" * 60)

best_name  = max(results, key=lambda n: results[n]["f1"])
best_model = results[best_name]["model"]

print(f"  Best Model  : {best_name}")
print(f"  Accuracy    : {results[best_name]['accuracy']:.4f}")
print(f"  Precision   : {results[best_name]['precision']:.4f}")
print(f"  Recall      : {results[best_name]['recall']:.4f}")
print(f"  F1 Score    : {results[best_name]['f1']:.4f}")


# ==============================================================
# STEP 7: FRAUD PREDICTION — FINAL OUTPUT
# ==============================================================
print("\n" + "=" * 60)
print("STEP 7: Fraud Prediction (Final Output)")
print("=" * 60)

def predict_transaction(model, transaction: pd.DataFrame) -> str:
    """
    Predict whether a single transaction is Fraud or Legitimate.

    Args:
        model       : Trained sklearn classifier
        transaction : Single-row DataFrame with same features as X

    Returns:
        'FRAUD' or 'LEGITIMATE'
    """
    prediction  = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0]
    label = "🚨 FRAUD" if prediction == 1 else " LEGITIMATE"
    print(f"  Result      : {label}")
    print(f"  Probability : Legitimate = {probability[0]:.4f} | Fraud = {probability[1]:.4f}")
    return label


print(f"\nUsing best model: {best_name}")
print("Running predictions on 5 random test samples...\n")

sample_indices = y_test.sample(5, random_state=7).index
for idx in sample_indices:
    sample = X_test.loc[[idx]]
    actual = "🚨 FRAUD" if y_test.loc[idx] == 1 else " LEGITIMATE"
    print(f"Transaction [{idx}]  |  Actual: {actual}")
    predict_transaction(best_model, sample)
    print()

print("=" * 60)
print("Pipeline complete.")
print(f"Best model '{best_name}' is ready for production use.")
print("=" * 60)
