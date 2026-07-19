# training/train_models.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score

# ======================
# Paths and Setup
# ======================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "stroke_dataset_wearable_case_based.csv"  # ✅ FIX: Match the new filename
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("📂 Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Drop id column if present
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# ======================
# ✅ FIX: Define Features EARLY
# ======================
numeric_features = [
    'age', 'avg_glucose_level', 'bmi',
    'hypertension', 'heart_disease',
    'physical_activity', 'sleep_hours',
    'alcohol_intake', 'diet_quality', 'family_stroke_history', 
    'heart_rate', 'spo2', 'steps'
]

categorical_features = [
    'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
]

# Separate target and features
y = df['stroke']
X = df[numeric_features + categorical_features] # Only keep columns we use

# ✅ FIX: Standardize all numeric columns to float
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# ✅ FIX: Ensure categorical columns are strings
for col in categorical_features:
    X[col] = X[col].astype(str)

# ======================
# Preprocessing
# ======================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# ======================
# Train-test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ======================
# Helper function (Keep your existing build_train_save)
# ======================
def build_train_save(clf, name, filename):
    print(f"\n🚀 Training {name} with SMOTE balancing...")
    
    pipe = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.6, k_neighbors=3)),
        ('clf', clf)
    ])

    if name == "Random Forest":
        # Simplified grid for faster training
        param_grid = {
            'clf__max_depth': [8, 10],
            'clf__n_estimators': [150, 200]
        }
        grid = GridSearchCV(pipe, param_grid, scoring='recall', cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        pipe = grid.best_estimator_
    else:
        pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

    print(f"🎯 ROC AUC: {auc}")
    print("📄 Classification report:\n", classification_report(y_test, y_pred))

    # Save
    joblib.dump(pipe, MODEL_DIR / filename)
    return pipe

# ======================
# Run Models (Keep your existing definitions)
# ======================
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
xgb = XGBClassifier(eval_metric='logloss', random_state=42, n_estimators=250, learning_rate=0.05)
lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')

rf_pipe = build_train_save(rf, "Random Forest", "rf_pipeline.pkl")
xgb_pipe = build_train_save(xgb, "XGBoost", "xgb_pipeline.pkl")
lr_pipe = build_train_save(lr, "Logistic Regression", "lr_pipeline.pkl")

print("\n✅ Training complete. Models saved with Heart Rate, SpO2, and Steps support.")