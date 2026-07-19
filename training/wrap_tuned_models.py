"""import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd

# Paths
MODEL_DIR = r"C:\Users\HP\Desktop\BrainStrokePrediction\model"
DATA_PATH = r"C:\Users\HP\Desktop\BrainStrokePrediction\dataset\stroke_data.csv"

# Load tuned models (already trained)
rf_tuned = joblib.load(f"{MODEL_DIR}\\rf_tuned.pkl")
xgb_tuned = joblib.load(f"{MODEL_DIR}\\xgb_tuned.pkl")

# Load a small sample of data to learn preprocessing schema
df = pd.read_csv(DATA_PATH)

# Drop ID column if present
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Separate target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Identify column types
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numeric_features = [col for col in X.columns if X[col].dtype != 'object']

# Build preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create and FIT pipelines using tuned models
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', rf_tuned)
])

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_tuned)
])

# Fit the pipelines (required so preprocessor learns schema)
print("🔹 Fitting wrapped pipelines...")
rf_pipeline.fit(X, y)
xgb_pipeline.fit(X, y)

# Save fitted pipelines
print("💾 Saving fitted pipelines...")
joblib.dump(rf_pipeline, f"{MODEL_DIR}\\rf_pipeline.pkl")
joblib.dump(xgb_pipeline, f"{MODEL_DIR}\\xgb_pipeline.pkl")

print("✅ Wrapped and fitted tuned models saved successfully!")
"""














import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import os

# ✅ Paths
MODEL_DIR = r"C:\Users\HP\Desktop\BrainStrokePrediction\model"
DATA_PATH = r"C:\Users\HP\Desktop\BrainStrokePrediction\dataset\stroke_dataset_New.csv"

# Ensure the model folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load tuned models (already trained)
print("🔹 Loading tuned models...")
rf_tuned = joblib.load(os.path.join(MODEL_DIR, "rf_tuned.pkl"))
xgb_tuned = joblib.load(os.path.join(MODEL_DIR, "xgb_tuned.pkl"))

# ✅ Load dataset to learn schema
print("🔹 Loading dataset for preprocessing schema...")
df = pd.read_csv(DATA_PATH)

# Drop ID column if present
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Separate target
X = df.drop('stroke', axis=1)
y = df['stroke']

# ✅ Identify categorical and numeric columns
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numeric_features = [col for col in X.columns if X[col].dtype != 'object']

print(f"🔹 Categorical features: {categorical_features}")
print(f"🔹 Numeric features: {numeric_features}")

# ✅ Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# ✅ Wrap tuned models in pipelines
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', rf_tuned)  # renamed from 'model' → 'clf' (more consistent naming)
])

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', xgb_tuned)
])

# ✅ Fit preprocessor schema (not retraining models)
print("🔹 Fitting preprocessing schema...")
rf_pipeline.named_steps['preprocessor'].fit(X)
xgb_pipeline.named_steps['preprocessor'].fit(X)

# ✅ Save fitted pipelines
print("💾 Saving fitted pipelines...")
joblib.dump(rf_pipeline, os.path.join(MODEL_DIR, "rf_pipeline.pkl"))
joblib.dump(xgb_pipeline, os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))

print("✅ Wrapped and fitted tuned models saved successfully!")
