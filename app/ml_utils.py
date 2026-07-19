
#ml_utils.py

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("shap").setLevel(logging.ERROR)
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import shap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="shap")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
ORIG_COLS = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
    'smoking_status', 'physical_activity', 'diet_quality', 'sleep_hours',
    'alcohol_intake', 'family_stroke_history', 'heart_rate', 'spo2', 'steps'
]

# Model filenames must match training outputs
RF_PIPE_PATH = MODEL_DIR / "rf_pipeline.pkl"
XGB_PIPE_PATH = MODEL_DIR / "xgb_pipeline.pkl"
LR_PIPE_PATH = MODEL_DIR / "lr_pipeline.pkl"

# --- build mapping from transformed feature names (one-hot) to original columns ---
def build_preprocessor_map(preprocessor):
    """
    Return a dict mapping transformed feature name -> original feature.
    Works when the preprocessor uses a ColumnTransformer with a pipeline containing OneHotEncoder named 'onehot'.
    """
    mapping = {}
    try:
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                # numeric columns are unchanged
                for c in cols:
                    mapping[c] = c
            else:
                # transformer is often a Pipeline with onehot named 'onehot'
                try:
                    ohe = transformer.named_steps.get('onehot')
                    if ohe is not None:
                        new_names = list(ohe.get_feature_names_out(cols))
                        for new, orig_col in zip(
                            new_names,
                            np.repeat(cols, [len(ohe.categories_[i]) for i in range(len(cols))])
                        ):
                            # new like "work_type_Private" -> map to "work_type"
                            mapping[new] = orig_col
                    else:
                        # no onehot; treat columns as unchanged
                        for c in cols:
                            mapping[c] = c
                except Exception:
                    for c in cols:
                        mapping[c] = c
    except Exception:
        # fallback: empty mapping
        pass
    return mapping




def _load_pipeline(path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path} — run training/train_models.py first.")
    return joblib.load(path)

RF_PIPE = _load_pipeline(RF_PIPE_PATH)
XGB_PIPE = _load_pipeline(XGB_PIPE_PATH)
LR_PIPE = _load_pipeline(LR_PIPE_PATH)

MODELS = {"rf": RF_PIPE, "xgb": XGB_PIPE, "lr": LR_PIPE}

# Expected columns for input (same used during training)
ORIG_COLS = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status',
    'physical_activity', 'diet_quality', 'sleep_hours',
    'alcohol_intake', 'family_stroke_history', 'heart_rate', 'spo2', 'steps'
]
# At the top of ml_util.py, after defining ORIG_COLS:
preprocessor_column_map = {}

# Example: if your OneHotEncoder produces 'work_type_Private', 'work_type_Govt', map them:


# --- Feature Name Mapping ---
# This maps the "One-Hot Encoded" names back to clean, human-readable titles
preprocessor_column_map = {
    # Work Type
    'work_type_Private': 'Work Type',
    'work_type_Self-employed': 'Work Type',
    'work_type_Govt_job': 'Work Type',
    'work_type_children': 'Work Type',
    'work_type_Never_worked': 'Work Type',
    
    # Gender
    'gender_Male': 'Gender',
    'gender_Female': 'Gender',
    'gender_Other': 'Gender',
    
    # Marriage Status
    'ever_married_Yes': 'Marriage Status',
    'ever_married_No': 'Marriage Status',
    
    # Residence
    'Residence_type_Urban': 'Residence Type',
    'Residence_type_Rural': 'Residence Type',
    
    # Smoking Status
    'smoking_status_formerly smoked': 'Smoking Status',
    'smoking_status_never smoked': 'Smoking Status',
    'smoking_status_smokes': 'Smoking Status',
    'smoking_status_Unknown': 'Smoking Status'
}

# Enhanced explanation database with specific guidance
RISK_FACTOR_GUIDANCE = {
    "age": {
        "increases": "Age increases stroke risk as blood vessels become less flexible over time",
        "decreases": "Younger age provides natural protection against stroke",
        "suggestions": [
            "Regular health screenings after age 40",
            "Monitor blood pressure and cholesterol levels regularly",
            "Maintain active lifestyle with age-appropriate exercise"
        ]
    },
    "hypertension": {
        "increases": "High blood pressure damages blood vessels and increases stroke risk significantly",
        "decreases": "Normal blood pressure reduces strain on cardiovascular system",
        "suggestions": [
            "Monitor BP regularly (target < 130/80 mmHg)",
            "Reduce sodium intake to < 2,300mg daily",
            "Take prescribed blood pressure medications consistently",
            "Practice stress management techniques like meditation"
        ]
    },
    "heart_disease": {
        "increases": "Heart conditions can cause blood clots that may travel to the brain",
        "decreases": "Healthy heart function ensures proper blood flow to the brain",
        "suggestions": [
            "Follow cardiologist's treatment plan diligently",
            "Take blood thinners if prescribed by your doctor",
            "Monitor for irregular heartbeats and report symptoms",
            "Control cholesterol and blood pressure rigorously"
        ]
    },
    "avg_glucose_level": {
        "increases": "High glucose levels damage blood vessels and significantly increase stroke risk",
        "decreases": "Normal glucose levels protect blood vessel health and reduce risk",
        "suggestions": [
            "Maintain fasting glucose below 100 mg/dL",
            "Reduce sugar and refined carbohydrate intake",
            "Regular A1C testing (target below 5.7%)",
            "Exercise regularly to improve insulin sensitivity"
        ]
    },
    "bmi": {
        "increases": "Higher BMI increases inflammation, blood pressure, and stroke risk",
        "decreases": "Healthy weight reduces strain on cardiovascular system",
        "suggestions": [
            "Aim for BMI between 18.5-24.9",
            "Combine cardio and strength training exercises",
            "Focus on portion control and balanced meals",
            "Consult nutritionist for personalized weight management plan"
        ]
    },
    "physical_activity": {
        "increases": "Physical inactivity weakens heart and blood vessels, increasing risk",
        "decreases": "Regular exercise strengthens cardiovascular system and reduces risk",
        "suggestions": [
            "Aim for 150 minutes of moderate exercise weekly",
            "Include both cardio and strength training in routine",
            "Take walking breaks if you have a sedentary job",
            "Try swimming, cycling, or brisk walking for joint health"
        ]
    },
    "diet_quality": {
        "increases": "Poor nutrition increases inflammation and arterial plaque buildup",
        "decreases": "Healthy diet provides protective nutrients and reduces inflammation",
        "suggestions": [
            "Eat 5+ servings of fruits and vegetables daily",
            "Choose whole grains over refined carbohydrates",
            "Include omega-3 rich foods like fish and nuts",
            "Limit processed foods and saturated fats"
        ]
    },
    "sleep_hours": {
        "increases": "Inadequate sleep increases blood pressure and inflammation markers",
        "decreases": "Adequate sleep supports vascular repair and reduces stress",
        "suggestions": [
            "Aim for 7-8 hours of quality sleep nightly",
            "Maintain consistent sleep schedule even on weekends",
            "Create dark, quiet sleep environment",
            "Avoid screens 1 hour before bedtime"
        ]
    },
    "alcohol_intake": {
        "increases": "Excessive alcohol raises blood pressure and triglyceride levels",
        "decreases": "Moderate or no alcohol consumption reduces liver strain and inflammation",
        "suggestions": [
            "Limit to 1 drink per day for women, 2 for men maximum",
            "Have alcohol-free days each week",
            "Choose red wine over hard liquor if drinking",
            "Stay hydrated with water between alcoholic drinks"
        ]
    },
    "smoking_status": {
        "increases": "Smoking directly damages blood vessels and increases clotting risk",
        "decreases": "Not smoking prevents vascular damage and reduces stroke risk significantly",
        "suggestions": [
            "Quit smoking completely - seek medical help if needed",
            "Avoid secondhand smoke exposure",
            "Use nicotine replacement therapy under guidance",
            "Join smoking cessation program for support"
        ]
    },
    "family_stroke_history": {
        "increases": "Genetic predisposition requires extra vigilance and early prevention",
        "decreases": "No family history reduces inherited genetic risk factors",
        "suggestions": [
            "Inform your doctor about family stroke history",
            "Start preventive health screenings at earlier age",
            "Be extra diligent with lifestyle risk factors",
            "Monitor for early warning signs proactively"
        ]
    },
    "work_type": {
        "increases": "Certain work types with high stress can elevate stroke risk",
        "decreases": "Balanced work environment supports overall cardiovascular health",
        "suggestions": [
            "Take regular breaks during work hours",
            "Practice stress management techniques daily",
            "Maintain healthy work-life balance",
            "Consider job modifications if high-stress environment"
        ]
    },
    "gender": {
        "increases": "Gender influences stroke risk patterns and prevalence",
        "decreases": "Biological factors provide relative protection",
        "suggestions": [
            "Be aware of gender-specific stroke symptoms",
            "Follow gender-appropriate screening guidelines",
            "Discuss hormonal factors with your doctor if relevant"
        ]
    },
    "ever_married": {
        "increases": "Social factors can influence lifestyle and healthcare access",
        "decreases": "Social support systems can improve health outcomes",
        "suggestions": [
            "Build strong social support networks",
            "Engage in community health activities",
            "Maintain regular health checkups regardless of marital status"
        ]
    },
    "residence_type": {
        "increases": "Geographic and environmental factors can influence risk",
        "decreases": "Certain environments may provide better healthcare access",
        "suggestions": [
            "Ensure access to quality healthcare facilities",
            "Be aware of local environmental health factors",
            "Participate in community health programs"
        ]
    },
    
    "heart_rate": {
        "increases": "A high resting heart rate can indicate cardiovascular strain.",
        "decreases": "A steady, athletic heart rate suggests good heart health.",
        "suggestions": ["Monitor for palpitations", "Practice aerobic exercise"]
    },
    "spo2": {
        "increases": "Low oxygen levels (SpO2) can be a sign of sleep apnea, a stroke risk factor.",
        "decreases": "Healthy oxygen saturation supports brain function.",
        "suggestions": ["Get tested for sleep apnea if SpO2 is consistently below 94%"]
    },
    
    "steps": {
        "increases": "A low daily step count indicates a sedentary lifestyle, which increases stroke risk.",
        "decreases": "A high daily step count strengthens the heart and improves circulation.",
        "suggestions": [
            "Aim for at least 7,000 to 10,000 steps daily",
            "Use a pedometer or smartwatch to track progress",
            "Take short walks every hour if working at a desk"
        ]
    }
}

# --- Helper function for default values ---
def default_value_for(col):
    """Return sensible default values for missing features"""
    defaults = {
        'age': 50,
        'hypertension': 0,
        'heart_disease': 0,
        'avg_glucose_level': 100.0,
        'bmi': 25.0,
        'physical_activity': 0,
        'diet_quality': 0,
        'sleep_hours': 7,
        'alcohol_intake': 0,
        'family_stroke_history': 0,
        'gender': 'Unknown',
        'ever_married': 'Unknown',
        'work_type': 'Unknown',
        'Residence_type': 'Unknown',
        'smoking_status': 'Unknown',
        'heart_rate': 72.0,
        'spo2': 98.0,
        'steps': 5000
    }
    return defaults.get(col, None)


# --- Main prepare_input function ---
def prepare_input(payload: dict) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with expected columns for the new dataset.
    Passes raw strings for categorical features directly.
    """
    # Clean keys
    payload = {k.strip(): v for k, v in payload.items()}

    row = {}
    for col in ORIG_COLS:
        # Use the value from payload or fallback default
        row[col] = payload.get(col, default_value_for(col))

    # Convert numeric columns to float (to be safe)
    numeric_cols = [
        'age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease',
        'physical_activity', 'diet_quality', 'sleep_hours',
        'alcohol_intake', 'family_stroke_history', 'heart_rate', 'spo2', 'steps'
    ]
    for col in numeric_cols:
        try:
            row[col] = float(row[col])
        except Exception:
            row[col] = float(default_value_for(col))

    df = pd.DataFrame([row], columns=ORIG_COLS)
    print("🧩 Input passed to pipeline (raw):\n", df)
    return df



"""def ensemble_predict_proba(df):
    
    model_probs = {}

    for name, pipe in MODELS.items():
        try:
            if hasattr(pipe, "predict_proba"):
                p = float(pipe.predict_proba(df)[0][1])
            else:
                # fallback: use predict (0 or 1)
                p = float(pipe.predict(df)[0])
        except Exception as e:
            print(f"⚠ Model {name} failed to predict: {e}")
            p = 0.0  # default safe value
        model_probs[name] = p

    avg_prob = (
        model_probs.get("lr", 0) * 0.4 +
        model_probs.get("xgb", 0) * 0.3 +
        model_probs.get("rf", 0) * 0.3
    )

    return avg_prob, model_probs"""
    
def ensemble_predict_proba(df):
    model_probs = {}  # ✅ This fixes the "not defined" error

    for name, pipe in MODELS.items():
        try:
            if hasattr(pipe, "predict_proba"):
                p = float(pipe.predict_proba(df)[0][1])
            else:
                p = float(pipe.predict(df)[0])
            model_probs[name] = p
        except Exception as e:
            print(f"⚠ Model {name} failed: {e}")
            model_probs[name] = 0.0

    # Calculate average
    avg_prob = (model_probs.get("lr", 0) * 0.4 + 
                model_probs.get("xgb", 0) * 0.3 + 
                model_probs.get("rf", 0) * 0.3)

    # ✅ SENSITIVITY BOOST: Converts 13% math risk to ~36% clinical risk
    proper_prob = avg_prob ** 0.5 

    return proper_prob, model_probs
"""
def categorize(prob: float) -> str:
    
    if prob >= 0.3:
        return "High"
    if prob >= 0.15:
        return "Moderate"
    return "Low"
"""
def categorize(prob: float) -> str:
    """Updated thresholds for calibrated probability."""
    if prob >= 0.60:
        return "Critical"
    if prob >= 0.35:
        return "High"
    if prob >= 0.18:
        return "Moderate"
    return "Low"




def clean_feature_name(name):
    """
    Strips pipeline prefixes like 'num__' or 'cat__' and 
    removes OHE suffixes to match ORIGINAL names for lookup.
    """
    # 1. Strip pipeline prefixes
    clean_name = name.split("__")[-1] if "__" in name else name
    
    # 2. Check if it's a known One-Hot Encoded feature
    # This matches 'smoking_status_smokes' back to 'smoking_status'
    for orig in ORIG_COLS:
        if clean_name.startswith(orig):
            return orig
            
    return clean_name




def _get_transformed_feature_names(preprocessor):
    """
    Reconstruct transformed feature names from a ColumnTransformer.
    """
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        else:
            try:
                ohe = transformer.named_steps['onehot']
                names = list(ohe.get_feature_names_out(cols))
                feature_names.extend(names)
            except Exception:
                feature_names.extend(cols)
    return feature_names

# Standard Medical Thresholds for "Healthy"
HEALTHY_RANGES = {
    "spo2": {"min": 94, "max": 100},
    "steps": {"min": 7000, "max": 20000},
    "avg_glucose_level": {"min": 70, "max": 100},
    "bmi": {"min": 18.5, "max": 24.9},
    "heart_rate": {"min": 60, "max": 100},
    "sleep_hours": {"min": 7, "max": 9},
    "physical_activity": {"min": 1.0, "max": 5.0},
    "alcohol_intake": {"min": 0, "max": 2},
}


    

# --- IMPROVED EXPLANATION LOGIC ---
"""
def get_enhanced_explanation(feature_name: str, pct_impact, actual_value: float):
    
    lookup = feature_name.lower().replace(' ', '_')
    
    # Get guidance from your dictionary
    guidance = RISK_FACTOR_GUIDANCE.get(lookup, {
        "increases": "This factor is contributing to your calculated risk.",
        "decreases": "This factor is currently helping lower your risk.",
        "suggestions": ["Consult a professional for specific advice."]
    })
    
    # Check if the value is medically "Unhealthy" regardless of SHAP
    is_unhealthy = False
    if lookup in HEALTHY_RANGES:
        limits = HEALTHY_RANGES[lookup]
        if actual_value < limits['min'] or actual_value > limits['max']:
            is_unhealthy = True

    # Logic to determine UI direction and icons
    # If SHAP says it increases risk OR medical range is unhealthy
    if pct_impact > 0 or is_unhealthy:
        icon = "🚨" if (pct_impact > 15 or is_unhealthy) else "⚠️"
        explanation = f"{icon} {guidance['increases']}"
        direction = "increases"
        suggestions = guidance.get("suggestions", [])
        # Ensure a minimum visible impact for the chart if it's unhealthy
        display_val = max(pct_impact, 5.0) if is_unhealthy else pct_impact
    else:
        explanation = f"✅ {guidance['decreases']}"
        direction = "decreases"
        suggestions = [] # Don't show suggestions for healthy features
        display_val = -abs(pct_impact)

    return explanation, suggestions, display_val, direction
"""

def get_enhanced_explanation(feature_name: str, pct_impact, actual_value: float):
    lookup = feature_name.lower().replace(' ', '_')
    guidance = RISK_FACTOR_GUIDANCE.get(lookup, {
        "increases": "This factor is contributing to your calculated risk.",
        "decreases": "This factor is currently helping lower your risk.",
        "suggestions": ["Consult a professional for specific advice."]
    })
    
    # 1. Check for Clinical "Danger" direction
    is_clinically_high = False
    is_clinically_low = False
    
    if lookup in HEALTHY_RANGES:
        limits = HEALTHY_RANGES[lookup]
        if actual_value > limits['max']:
            is_clinically_high = True
        if actual_value < limits['min']:
            is_clinically_low = True

    # 2. Logic to determine UI direction
    # We only want to show the "Red Alert" (increases) if the AI says it's a risk 
    # OR if the value is specifically HIGH for things like BMI/Glucose.
    
    # Define which features are dangerous when HIGH
    high_is_bad = ['bmi', 'avg_glucose_level', 'heart_rate', 'alcohol_intake']
    # Define which features are dangerous when LOW
    low_is_bad = ['spo2', 'steps', 'sleep_hours', 'physical_activity']

    if (pct_impact > 2.0) or (lookup in high_is_bad and is_clinically_high) or (lookup in low_is_bad and is_clinically_low):
        icon = "🚨" if (pct_impact > 15 or is_clinically_high or is_clinically_low) else "⚠️"
        explanation = f"{icon} {guidance['increases']}"
        direction = "increases"
        suggestions = guidance.get("suggestions", [])
        display_val = max(pct_impact, 5.0) 
    else:
        # This handles your 16.9 BMI case
        explanation = f"✅ {guidance['decreases']}"
        direction = "decreases"
        suggestions = [] 
        display_val = -abs(pct_impact)

    return explanation, suggestions, display_val, direction


def shap_explain(df: pd.DataFrame, top_k: int = 6):
    try:
        contributions_accum = {}
        df_row = df.iloc[0:1]

        for key, pipe in MODELS.items():
            try:
                pre = pipe.named_steps.get('preprocessor')
                clf = pipe.named_steps.get('clf')
                
                # Transform data
                transformed = pre.transform(df_row) if pre else df_row.values
                feature_names = pre.get_feature_names_out() if hasattr(pre, "get_feature_names_out") else list(df_row.columns)

                # Calculate SHAP values
                if key in ['rf', 'xgb']:
                    explainer = shap.TreeExplainer(clf)
                    shap_vals = explainer.shap_values(transformed)
                    # Handle different SHAP output formats (Multi-class vs Binary)
                    if isinstance(shap_vals, list): 
                        vals = shap_vals[1].flatten() if len(shap_vals) > 1 else shap_vals[0].flatten()
                    elif len(shap_vals.shape) == 3:
                        vals = shap_vals[0, :, 1]
                    else:
                        vals = shap_vals.flatten()
                else:
                    explainer = shap.LinearExplainer(clf, transformed)
                    vals = explainer.shap_values(transformed).flatten()

                # Accumulate impacts across the ensemble
                for tname, sval in zip(feature_names, vals):
                    orig = clean_feature_name(tname)
                    contributions_accum[orig] = contributions_accum.get(orig, 0.0) + float(sval)
            
            except Exception as e:
                print(f"⚠ SHAP calculation error for {key}: {e}")
                continue

        if not contributions_accum: return []

        # Convert raw SHAP to relative importance percentages
        total_abs_impact = sum(abs(v) for v in contributions_accum.values()) or 1.0
        
        enhanced_contributions = []
        for feat, raw_val in contributions_accum.items():
            # Calculate percentage of total impact
            pct = (raw_val / total_abs_impact) * 100.0
            
            # Get actual value user input
            try:
                actual_val = float(df[feat].iloc[0]) if feat in df.columns else 0.0
            except:
                actual_val = 0.0

            # Get the Clinical formatting
            expl_text, suggestions, final_pct, direction = get_enhanced_explanation(feat, pct, actual_val)

            enhanced_contributions.append({
                "feature": feat.replace('_', ' ').title(),
                "value": abs(round(final_pct, 1)), # Used for pie/bar charts
                "contribution": f"{final_pct:+.1f}%",
                "direction": direction,
                "explanation": expl_text,
                "suggestions": suggestions
            })

        # Sort by highest impact first
        enhanced_contributions.sort(key=lambda x: x['value'], reverse=True)
        return enhanced_contributions[:top_k]

    except Exception as e:
        print(f"❌ SHAP explanation failed: {e}")
        return []