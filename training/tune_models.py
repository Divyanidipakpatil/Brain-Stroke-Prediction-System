# ✅ 1️⃣ Import Libraries
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ✅ 2️⃣ Load and Prepare Data
print("🔹 Loading and preparing dataset...")

# Load dataset
df = pd.read_csv(r"C:\Users\HP\Desktop\BrainStrokePrediction\dataset\stroke_dataset_New.csv")


# Drop unnecessary columns if any (like id)
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Handle missing values before SMOTE
print("🔹 Handling missing values...")
X = X.fillna(X.median())  # Fill numeric NaNs with median values
print("🔹 Applying SMOTE to handle imbalance...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ✅ 3️⃣ Tune RandomForest with GridSearchCV
print("\n🔹 Tuning Random Forest... (this may take a few minutes)")
rf = RandomForestClassifier(random_state=42)

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

rf_grid.fit(X_train, y_train)

print("\n✅ Best Random Forest Parameters:", rf_grid.best_params_)
print("✅ Best RF ROC AUC (CV):", rf_grid.best_score_)

rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)

print("\n🔸 Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("🔸 Random Forest Test ROC AUC:", roc_auc_score(y_test, rf_pred))

# ✅ 4️⃣ Tune XGBoost with GridSearchCV
print("\n🔹 Tuning XGBoost... (this may take a while)")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}

xgb_grid = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_params,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

xgb_grid.fit(X_train, y_train)

print("\n✅ Best XGBoost Parameters:", xgb_grid.best_params_)
print("✅ Best XGB ROC AUC (CV):", xgb_grid.best_score_)

xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test)

print("\n🔸 XGBoost Classification Report:\n", classification_report(y_test, xgb_pred))
print("🔸 XGBoost Test ROC AUC:", roc_auc_score(y_test, xgb_pred))

# ✅ 5️⃣ Save Tuned Models
print("\n💾 Saving tuned models...")
joblib.dump(rf_best, r"C:\Users\HP\Desktop\BrainStrokePrediction\model\rf_tuned.pkl")
joblib.dump(xgb_best, r"C:\Users\HP\Desktop\BrainStrokePrediction\model\xgb_tuned.pkl")

print("\n🎯 Tuning Complete! Best models saved successfully.")
"""













# ✅ 1️⃣ Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# ✅ 2️⃣ Load and Prepare Data
print("🔹 Loading and preparing dataset...")

# Load dataset
df = pd.read_csv(r"C:\Users\HP\Desktop\BrainStrokePrediction\dataset\stroke_dataset_New.csv")

# Drop unnecessary columns if any (like id)
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Encode categorical features (convert non-numeric into numeric)
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Handle missing values before SMOTE
print("🔹 Handling missing values...")
X = X.fillna(X.median())  # Fill numeric NaNs with median values

# Apply SMOTE for balancing
print("🔹 Applying SMOTE to handle imbalance...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ✅ 3️⃣ Random Forest (Optimized)
print("\n🔹 Tuning Random Forest... (optimized for speed & recall)")
rf = RandomForestClassifier(random_state=42)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=3,  # reduced from 5 → faster tuning
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

rf_grid.fit(X_train, y_train)

print("\n✅ Best Random Forest Parameters:", rf_grid.best_params_)
print("✅ Best RF ROC AUC (CV):", rf_grid.best_score_)

rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)

print("\n🔸 Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("🔸 Random Forest Test ROC AUC:", roc_auc_score(y_test, rf_pred))

# ✅ 4️⃣ XGBoost (Optimized)
print("\n🔹 Tuning XGBoost... (optimized for speed & accuracy)")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

xgb_grid = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_params,
    cv=3,  # reduced for speed
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

xgb_grid.fit(X_train, y_train)

print("\n✅ Best XGBoost Parameters:", xgb_grid.best_params_)
print("✅ Best XGB ROC AUC (CV):", xgb_grid.best_score_)

xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test)

print("\n🔸 XGBoost Classification Report:\n", classification_report(y_test, xgb_pred))
print("🔸 XGBoost Test ROC AUC:", roc_auc_score(y_test, xgb_pred))

# ✅ 5️⃣ Save Tuned Models and Scaler
print("\n💾 Saving tuned models...")

# Ensure model folder exists
os.makedirs(r"C:\Users\HP\Desktop\BrainStrokePrediction\model", exist_ok=True)

joblib.dump(rf_best, r"C:\Users\HP\Desktop\BrainStrokePrediction\model\rf_tuned.pkl")
joblib.dump(xgb_best, r"C:\Users\HP\Desktop\BrainStrokePrediction\model\xgb_tuned.pkl")
joblib.dump(scaler, r"C:\Users\HP\Desktop\BrainStrokePrediction\model\scaler.pkl")

print("\n🎯 Tuning Complete! Best models and scaler saved successfully.")
