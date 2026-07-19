
"""
# ==============================
# 📦 IMPORT REQUIRED LIBRARIES
# ==============================
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from pathlib import Path

# ==============================
# 📁 DEFINE FILE PATHS
# ==============================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "stroke_data.csv"
MODEL_DIR = ROOT / "model"

# ==============================
# 📂 LOAD AND PREPARE DATA
# ==============================
print(f"📂 Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['stroke'])  # ensure target exists

# Split into features (X) and target (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# Stratified split ensures class balance in test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 🤖 LOAD TRAINED MODELS
# ==============================
models = {
    "Random Forest": joblib.load(MODEL_DIR / "rf_pipeline.pkl"),
    "XGBoost": joblib.load(MODEL_DIR / "xgb_pipeline.pkl"),
    "Logistic Regression": joblib.load(MODEL_DIR / "lr_pipeline.pkl"),
}

# Dictionaries for metrics
train_accuracies = {}
test_accuracies = {}
roc_data = {}

# ==============================
# 🧮 MODEL EVALUATION
# ==============================
print("\n===== 🧠 MODEL EVALUATION RESULTS =====\n")

for name, model in models.items():
    print(f"🔹 Evaluating {name}...\n")

    # ✅ TRAINING ACCURACY
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # ✅ TESTING ACCURACY
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_accuracies[name] = train_acc
    test_accuracies[name] = test_acc

    # 📈 ROC Curve + AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)

    # 📊 CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"✅ Training Accuracy: {train_acc:.3f}")
    print(f"✅ Testing Accuracy:  {test_acc:.3f}")
    print(f"📈 ROC-AUC Score: {auc:.3f}\n" if 'auc' in locals() else "")

    # 🧾 Print Classification Report
    print("📋 Confusion Matrix:")
    print(cm)
    print(f"""
    #True Negatives (TN): {tn} → Correctly predicted NO stroke
    #False Positives (FP): {fp} → Predicted stroke but actually NO stroke
    #False Negatives (FN): {fn} → Missed cases (actual stroke but predicted NO stroke)
    #True Positives (TP): {tp} → Correctly predicted stroke
""")
    print("📄 Classification Report:\n", classification_report(y_test, y_test_pred, digits=3))
    print("------------------------------------------------------------")

    # 🎨 Visual Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"🧩 Confusion Matrix - {name}", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.xticks([0.5, 1.5], ["No Stroke", "Stroke"])
    plt.yticks([0.5, 1.5], ["No Stroke", "Stroke"])
    plt.tight_layout()
    plt.show()

# ==============================
# 📊 VISUALIZATION 1:
# Training vs Testing Accuracy
# ==============================
plt.figure(figsize=(9, 5))
bar_width = 0.35
models_list = list(models.keys())
train_values = [train_accuracies[m] for m in models_list]
test_values = [test_accuracies[m] for m in models_list]

# Create side-by-side bars
plt.bar(
    [x for x in range(len(models_list))],
    train_values, width=bar_width, label="Training Accuracy", color="#4C72B0", alpha=0.85
)
plt.bar(
    [x + bar_width for x in range(len(models_list))],
    test_values, width=bar_width, label="Testing Accuracy", color="#55A868", alpha=0.85
)

plt.xlabel("Machine Learning Models", fontsize=11)
plt.ylabel("Accuracy Score", fontsize=11)
plt.title("📊 Training vs Testing Accuracy Comparison", fontsize=13, fontweight='bold')
plt.xticks([x + bar_width / 2 for x in range(len(models_list))], models_list)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ==============================
# 📈 VISUALIZATION 2:
# ROC Curves Comparison
# ==============================
plt.figure(figsize=(8, 6))

for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Guess (AUC = 0.5)")
plt.title("📈 ROC Curves for Stroke Prediction Models", fontsize=13, fontweight='bold')
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

print("\n✅ Model evaluation and visualization completed successfully!")
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ==============================
# 📦 IMPORT REQUIRED LIBRARIES
# ==============================
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from pathlib import Path

# ==============================
# 📁 DEFINE FILE PATHS
# ==============================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "stroke_dataset_New.csv"
MODEL_DIR = ROOT / "model"

# ==============================
# 📂 LOAD AND PREPARE DATA
# ==============================
print(f"📂 Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['stroke'])  # ensure target exists

# Split into features (X) and target (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# Stratified split ensures class balance in test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 🤖 LOAD TRAINED MODELS
# ==============================
models = {
    "Random Forest": joblib.load(MODEL_DIR / "rf_pipeline.pkl"),
    "XGBoost": joblib.load(MODEL_DIR / "xgb_pipeline.pkl"),
    "Logistic Regression": joblib.load(MODEL_DIR / "lr_pipeline.pkl"),
}

# Dictionaries for metrics
train_accuracies = {}
test_accuracies = {}
roc_data = {}

# ==============================
# 🧮 MODEL EVALUATION
# ==============================
print("\n===== 🧠 MODEL EVALUATION RESULTS =====\n")

for name, model in models.items():
    print(f"🔹 Evaluating {name}...\n")

    # ✅ TRAINING ACCURACY
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # ✅ TESTING ACCURACY
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_accuracies[name] = train_acc
    test_accuracies[name] = test_acc

    # 📈 ROC Curve + AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)

    # 📊 CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"✅ Training Accuracy: {train_acc:.3f}")
    print(f"✅ Testing Accuracy:  {test_acc:.3f}")
    print(f"📈 ROC-AUC Score: {auc:.3f}\n" if 'auc' in locals() else "")

    # 🧾 Print Classification Report
    print("📋 Confusion Matrix:")
    print(cm)
    print(f"""
    True Negatives (TN): {tn} → Correctly predicted NO stroke
    False Positives (FP): {fp} → Predicted stroke but actually NO stroke
    False Negatives (FN): {fn} → Missed cases (actual stroke but predicted NO stroke)
    True Positives (TP): {tp} → Correctly predicted stroke
    """)
    print("📄 Classification Report:\n", classification_report(y_test, y_test_pred, digits=3))
    print("------------------------------------------------------------")

    # 🎨 Visual Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"🧩 Confusion Matrix - {name}", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.xticks([0.5, 1.5], ["No Stroke", "Stroke"])
    plt.yticks([0.5, 1.5], ["No Stroke", "Stroke"])
    plt.tight_layout()
    plt.show()

# ==============================
# 📊 VISUALIZATION 1:
# Training vs Testing Accuracy
# ==============================
plt.figure(figsize=(9, 5))
bar_width = 0.35
models_list = list(models.keys())
train_values = [train_accuracies[m] for m in models_list]
test_values = [test_accuracies[m] for m in models_list]

# Create side-by-side bars
plt.bar(
    [x for x in range(len(models_list))],
    train_values, width=bar_width, label="Training Accuracy", color="#4C72B0", alpha=0.85
)
plt.bar(
    [x + bar_width for x in range(len(models_list))],
    test_values, width=bar_width, label="Testing Accuracy", color="#55A868", alpha=0.85
)

plt.xlabel("Machine Learning Models", fontsize=11)
plt.ylabel("Accuracy Score", fontsize=11)
plt.title("📊 Training vs Testing Accuracy Comparison", fontsize=13, fontweight='bold')
plt.xticks([x + bar_width / 2 for x in range(len(models_list))], models_list)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ==============================
# 📈 VISUALIZATION 2:
# ROC Curves Comparison
# ==============================
plt.figure(figsize=(8, 6))

for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Guess (AUC = 0.5)")
plt.title("📈 ROC Curves for Stroke Prediction Models", fontsize=13, fontweight='bold')
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

print("\n✅ Model evaluation and visualization completed successfully!")