# app/main.py

import os
import sys

# १. ही ओळ पायथनला सांगते की 'app' फोल्डरमध्येच फाईल्स शोधा
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# २. आता तुम्ही तुमची जुनी कमांड (विना डॉट वाली) वापरू शकता
# main.py madhe he badal kara:
try:
    # send_stroke_alert la 'as send_alert_to_doctor' mhanun import kara
    from sms_handler import send_stroke_alert as send_alert_to_doctor
except ImportError:
    from .sms_handler import send_stroke_alert as send_alert_to_doctor

from flask import Flask, request, jsonify, render_template, redirect, session, url_for
from flask_cors import CORS
# 'TrustedContact' इथे ॲड करा
from models import db, User, PredictionRecord, TrustedContact, UserProfile
from ml_utils import prepare_input, ensemble_predict_proba, categorize, shap_explain
from pathlib import Path 
import json
from alerts import send_doctor_report
from datetime import datetime
from flask import render_template

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

#

BASE_DIR = Path(__file__).resolve().parents[1]
db_path = BASE_DIR / "app.db"
 #if os.path.exists(db_path):
     #try:
        # सर्व्हिस सुरू होण्यापूर्वी जुना डेटाबेस साफ करणे
        #os.remove(db_path)
       # print("--- SUCCESS: Old database deleted to update columns! ---")
      #except Exception as e:
       # print(f"--- WARNING: Close other apps using app.db: {e} ---")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + str(BASE_DIR / "app.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "secret-key"
db.init_app(app)

with app.app_context():
    db.create_all()


# -------------------------------
# HOME
# -------------------------------
#@app.route("/")
#def index():
 #   if 'user_id' not in session:
  #      return redirect(url_for('login'))
   # return render_template("index.html")
@app.route("/", methods=['GET', 'POST']) 
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # 1. Form madhun data gya
        # example: age = request.form.get('age')
        
        # 2. Prediction logic call kara (tumcha model)
        
        # 3. Result sobat page parat dakhva
        return render_template("index.html", prediction_text="High/Low Risk result ithe dakhva")

    # Jar normal page visit asel (GET) tar:
    return render_template("index.html")


# -------------------------------
# REGISTER
# -------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        existing = User.query.filter_by(email=email).first()
        if existing:
            return render_template("register.html", error="Email already exists")

        user = User(name=name, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template("register.html")


# -------------------------------
# LOGIN
# -------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session['user_id'] = user.id
            session['user_name'] = user.name
            return redirect(url_for('index'))

        return render_template("login.html", error="Invalid credentials")
    
    return render_template("login.html")


# -------------------------------
# LOGOUT
# -------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))


# -------------------------------
# PREDICT API
# -------------------------------
"""

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "Invalid JSON"}), 400

        print("----------------------------")
        print("DATA RECEIVED:", payload)
        print("----------------------------")

        # --- 1. USER ID & SESSION HANDLER ---
        if 'user_id' in session:
            current_user_id = session['user_id']
        elif payload.get('user_id'):
            current_user_id = payload.get('user_id')
        else:
            current_user_id = 1  # Default ID for testing
            print("Using Default User ID: 1")

        # --- 2. DATA CLEANING & BMI CALCULATION ---
        try:
            # Limits set for medical accuracy
            payload['age'] = min(max(float(payload.get('age', 40)), 0), 82)
            payload['avg_glucose_level'] = min(max(float(payload.get('avg_glucose_level', 100)), 55), 271)
            
            height_cm = float(payload.get('height', 170))
            weight_kg = float(payload.get('weight', 70))
            
            # BMI Formula: weight (kg) / [height (m)]^2
            bmi = round(weight_kg / ((height_cm / 100) ** 2), 1) if height_cm > 0 else 25.0
            payload['bmi'] = min(max(bmi, 10), 97)
        except (ValueError, ZeroDivisionError):
            return jsonify({"error": "Invalid numerical input (Age/Weight/Height)"}), 400

        # --- 3. PREDICTION ENGINE ---
        df = prepare_input(payload)
        prob, model_probs = ensemble_predict_proba(df)
        category = categorize(prob)

        # --- 4. ALERT LOGIC (SMS & EMAIL) ---
        clean_category = category.strip().lower()
        print(f"DEBUG: Category -> [{clean_category}], Prob -> {prob}")

        # High Risk detect hone par alert jayega
        if "high" in clean_category or "critical" in clean_category:
            print("🚨 URGENT: High Risk Detected! Sending Alerts...")

            # --- PART A: SMS ALERT ---
            try:
                # Default verified number (Twilio)
                target_no = "+918459881450" 
                
                # DB se number fetch karne ki koshish
                try:
                    current_user = User.query.get(current_user_id)
                    if current_user:
                        db_no = getattr(current_user, 'emergency_phone', getattr(current_user, 'phone', None))
                        if db_no:
                            target_no = db_no
                except:
                    print("Could not fetch phone from DB, using default.")

                # SMS bhej rahe hain
                send_alert_to_doctor(prob, category, target_no)
                print(f"✅ SMS sent to: {target_no}")
            except Exception as e:
                print(f"❌ SMS System Error: {e}")

            # --- PART B: EMAIL ALERT ---
            try:
                patient_name = payload.get('name', 'Valuable Patient') 
                doctor_email = "divyanipatil60@gmail.com" 
                
                send_doctor_report(category, patient_name, doctor_email)
                print("✅ Email Report sent successfully!")
            except Exception as e:
                print(f"❌ Email System Error: {e}")
        else:
            print(f"ℹ️ Low Risk ({clean_category}). Alerts skipped.")

        # --- 5. EXPLAINABILITY (SHAP) ---
        try:
            contributions = shap_explain(df, top_k=6)
        except Exception as e:
            print(f"⚠ SHAP Error: {e}")
            contributions = []

        # --- 6. FINAL RESPONSE ---
        return jsonify({
            "probability": prob,
            "category": category,
            "model_probabilities": model_probs,
            "contributions": contributions,
            "bmi": payload['bmi'],
            "status": "success"
        })

    except Exception as e:
        print(f"💥 CRITICAL ROUTE ERROR: {e}")
        return jsonify({"error": str(e)}), 500
"""




@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "Invalid JSON"}), 400

        # --- 1. USER ID & DATA CLEANING ---
        current_user_id = payload.get('user_id', 1)
        payload['age'] = min(max(float(payload.get('age', 40)), 0), 82)
        payload['avg_glucose_level'] = min(max(float(payload.get('avg_glucose_level', 100)), 55), 271)
        
        height_cm = float(payload.get('height', 170))
        weight_kg = float(payload.get('weight', 70))
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 1) if height_cm > 0 else 25.0
        payload['bmi'] = min(max(bmi, 10), 97)

        # --- 2. PREDICTION ENGINE ---
        df = prepare_input(payload)
        prob, model_probs = ensemble_predict_proba(df)
        category = categorize(prob)

        # --- 3. EXPLAINABILITY ---
        try:
            contributions = shap_explain(df, top_k=6)
        except Exception as e:
            print(f"⚠ SHAP Error: {e}")
            contributions = []

        # --- 4. DATABASE SAVING ---
        # Pehle record save karein taaki ID mil sake
        new_record = PredictionRecord(
            user_id=current_user_id,
            probability=prob,
            category=category,
            input_json=json.dumps(payload),
            explanation_json=json.dumps(contributions)
        )
        db.session.add(new_record)
        db.session.commit()

        # --- 5. ALERT LOGIC ---
        # SMS Alert
        try:
            target_no = "+918459881450" 
            send_alert_to_doctor(prob, category, target_no)
        except Exception as e:
            print(f"❌ SMS System Error: {e}")

        # Email Alert
        try:
            patient_name = payload.get('name', 'Valuable Patient')
            receiver_email = "divyanipatil60@gmail.com"
            
            raw_glucose = payload.get('avg_glucose_level')
            glucose_val = float(raw_glucose) if raw_glucose and str(raw_glucose).strip() else 100.0
            raw_hr = payload.get('heart_rate')
            hr_val = float(raw_hr) if raw_hr and str(raw_hr).strip() else 75.0

            real_radar = [
                min((glucose_val / 250) * 100, 100), 
                min((hr_val / 150) * 100, 100),         
                min((payload['bmi'] / 40) * 100, 100),                               
                95, 80, 70, 85 
            ]
            
            real_analysis = [
                {
                    "param": "Blood Glucose", 
                    "val": f"{glucose_val} mg/dL",
                    "status": "High" if glucose_val > 140 else "Normal",
                    "color": "#ef4444" if glucose_val > 140 else "#22c55e",
                    "sug": "Monitor sugar intake."
                },
                {
                    "param": "Body Mass Index (BMI)", 
                    "val": payload['bmi'],
                    "status": "Overweight" if payload['bmi'] > 25 else "Healthy",
                    "color": "#f59e0b" if payload['bmi'] > 25 else "#22c55e",
                    "sug": "Maintain active lifestyle."
                }
            ]

            rendered_html = render_template(
                "detailed_report.html", 
                record=new_record, 
                inputs=payload,
                advice="Maintain a healthy lifestyle.",
                contributions=contributions,
                radar_data=real_radar,
                trend_dates=[datetime.now().strftime("%d %b")], 
                trend_scores=[round(prob * 100, 1)],
                parameter_analysis=real_analysis,
                live_advice=[] 
            )
            
            
            
            send_doctor_report(category, patient_name, receiver_email, html_content=rendered_html)
            print(f"✅ Professional Email sent for Record ID: {new_record.id}")




        except Exception as e:
            print(f"❌ Email/Report Error: {e}")

        # --- 6. FINAL RESPONSE ---
        return jsonify({
            "probability": prob,
            "category": category,
            "model_probabilities": model_probs,
            "contributions": contributions,
            "bmi": payload['bmi'],
            "record_id": new_record.id,
            "status": "success"
        })

    except Exception as e:
        db.session.rollback()
        print(f"💥 CRITICAL ROUTE ERROR: {e}")
        return jsonify({"error": str(e)}), 500
 
       
# -------------------------------
# HISTORY
# -------------------------------
# @app.route("/history")
# def history():
#     if 'user_id' not in session:
#         return redirect(url_for('login'))

#     records = PredictionRecord.query.filter_by(
#         user_id=session['user_id']
#     ).order_by(PredictionRecord.created_at.desc()).all()

#     return render_template("history.html", records=records, user_name=session['user_name'])



@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)

    records = PredictionRecord.query.filter_by(user_id=user_id)\
        .order_by(PredictionRecord.created_at.desc()).all()

    import json

    for rec in records:
        try:
            rec.explanation_data = json.loads(rec.explanation_json)
        except:
            rec.explanation_data = []

    return render_template(
        'history.html',
        records=records,
        user_name=user.name
    )

@app.route("/vitals")
def vitals_tracking():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # १. डॅशबोर्डसाठी मागील ७ रेकॉर्ड्स मिळवणे (Weekly Trend साठी)
    history = PredictionRecord.query.filter_by(user_id=session['user_id'])\
        .order_by(PredictionRecord.created_at.desc())\
        .limit(7).all()
    
    history = history[::-1] # डेटा क्रमाने लावण्यासाठी (Mon to Sun)

    # २. ग्राफसाठी लिस्ट तयार करणे
    dates = [h.created_at.strftime('%a') for h in history] # Mon, Tue...
    hr_data = [json.loads(h.input_json).get('heart_rate', 0) for h in history]
    glucose_data = [json.loads(h.input_json).get('avg_glucose_level', 0) for h in history]
    
    # ३. लेटेस्ट माहिती मीटर आणि कार्ड्ससाठी
    latest_record = history[-1] if history else None
    latest_vitals = json.loads(latest_record.input_json) if latest_record else {}
    prob = latest_record.probability * 100 if latest_record else 0
    category = latest_record.category if latest_record else "No Data"

    return render_template("vitals.html", 
                           vitals=latest_vitals, 
                           dates=dates, 
                           hr_data=hr_data, 
                           gl_data=glucose_data,
                           probability=round(prob, 1),
                           category=category)
# ==========================================


# ... (Keep your imports and initial app config/db init the same) ...

# -------------------------------
# PROFILE
# -------------------------------
# -------------------------------
# PROFILE (Updated with Database Logic)
# -------------------------------
# @app.route("/profile")
# def profile():
#     # १. युजर लॉगिन आहे की नाही ते तपासा
#     if 'user_id' not in session:
#         return redirect(url_for('login'))

#     # २. डेटाबेसमधून 'User' मॉडेल वापरून माहिती काढा
#     user = User.query.get(session['user_id'])

#     if user:
#         # ३. युजरने किती प्रेडिक्शन्स केली आहेत त्याची संख्या मोजा (Stats Box साठी)
#         prediction_count = PredictionRecord.query.filter_by(user_id=user.id).count()

#         # ४. ही माहिती HTML कडे पाठवा
#         # टीप: तुमच्या 'User' मॉडेलमध्ये 'name' आणि 'email' हे कॉलम आहेत
#         return render_template("profile.html", 
#                                user=user,
#                                name=user.name, 
#                                email=user.email,
#                                prediction_count=prediction_count,
#                                member_since=user.created_at.strftime('%B %Y') if hasattr(user, 'created_at') else "March 2026")
    
#     return "User not found", 404

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # 1. Database se objects uthayein
    user_obj = User.query.get(session['user_id'])
    profile_data = UserProfile.query.filter_by(user_id=user_obj.id).first()
    contact = TrustedContact.query.filter_by(user_id=user_obj.id).first()
    
    # 2. HTML ko 'user' variable dena zaroori hai (UndefinedError hatane ke liye)
    # Hum 'user_obj' ko 'user' naam se bhej rahe hain
    return render_template('profile.html', 
                           user=user_obj,  # <--- Ye line UndefinedError fix karegi
                           name=user_obj.name, 
                           email=user_obj.email,
                           address=profile_data.address if (profile_data and hasattr(profile_data, 'address')) else "",
                           phone=getattr(user_obj, 'phone', ""), 
                           emergency_phone=contact.phone if contact else "",
                           emergency_email=contact.email if (contact and hasattr(contact, 'email')) else "")

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if user:
        try:
            # 1. Basic Info Update
            user.name = request.form.get('name')
            # --- NEW: Phone number save logic ---
            user.phone = request.form.get('phone') 
            
            # 2. UserProfile (Address) Update
            up = UserProfile.query.filter_by(user_id=user.id).first()
            if not up:
                up = UserProfile(user_id=user.id, age=20.0) 
                db.session.add(up)
            
            if hasattr(up, 'address'):
                up.address = request.form.get('address')
            
            # 3. Emergency Contact (Phone & Email) Update
            tc = TrustedContact.query.filter_by(user_id=user.id).first()
            if not tc:
                tc = TrustedContact(user_id=user.id, name="Emergency Contact")
                db.session.add(tc)
            
            tc.phone = request.form.get('emergency_phone')
            # --- NEW: Emergency Email save logic ---
            # Note: Ensure your TrustedContact model has 'email' column
            if hasattr(tc, 'email'):
                tc.email = request.form.get('emergency_email')

            db.session.commit()
            return redirect(url_for('profile', success=1))
        except Exception as e:
            db.session.rollback()
            return f"Error: {str(e)}", 500
    return redirect(url_for('profile'))




@app.route('/health-plan')
def health_plan():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    user_id = session['user_id']
    
    # 1. Fetch User and Profile Data
    user = User.query.get(user_id)
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    
    # 2. Logic to handle missing data (Defaults)
    # This prevents the page from crashing if a new user hasn't filled their data yet
    health_data = {
    "bmi": float(profile.bmi) if (profile and profile.bmi) else 0,
    "avg_glucose_level": float(profile.avg_glucose_level) if (profile and profile.avg_glucose_level) else 0,
    "heart_rate": float(profile.heart_rate) if (profile and profile.heart_rate) else 0,
    "steps": int(profile.steps) if (profile and profile.steps) else 0,
    "sleep_hours": float(profile.sleep_hours) if (profile and profile.sleep_hours) else 0,
    "alcohol_intake": profile.alcohol_intake if (profile and profile.alcohol_intake) else "None",
    "spo2": profile.spo2 if (profile and profile.spo2) else "N/A",
    "age": profile.age if (profile and profile.age) else 0
}

    # 3. Render the English page with the data
    return render_template('health_tips.html', **health_data)




@app.route('/emergency_settings')
def emergency_settings():
    if 'user_id' in session:
        current_user_id = session['user_id']
        user = User.query.get(current_user_id)
        # समजा तुमच्या डेटाबेसमध्ये युजरचे नाव आणि फोन आहे
         #user = User.query.get(user_id)
        if not user:
            return "User not found", 404
        # २. या युजरने सेव्ह केलेले सर्व Trusted Contacts डेटाबेसमधून मिळवा
        contacts = TrustedContact.query.filter_by(user_id=current_user_id).all()
       # Ithe 'contacts' vapra (contacts_list nahi)
        if contacts:
            phone_numbers_string = ",".join([str(c.phone) for c in contacts])
        else:
            phone_numbers_string = ""
        
        # ३. दोन्ही गोष्टी 'emergency.html' ला पाठवा
        return render_template('emergency.html', user=user, contacts=contacts,all_numbers=phone_numbers_string, 
                               user_id=current_user_id,)
    return redirect(url_for('login'))



@app.route('/save_emergency_contact', methods=['POST'])
def save_emergency_contact():
    data = request.json
    # समजा सध्या लॉगिन असलेल्या युजरचा ID 1 आहे (तुमच्या Auth नुसार बदला)
    current_user_id = 1 

    name = data.get('name')
    phone = data.get('phone')
    to=phone

    # 1. इथे SMS पाठवण्याचे लॉजिक (Twilio) टाका
    # जर मेसेज गेला तर status = "Sent", नाहीतर "Failed"
    sms_status = "Sent" 

    # 2. डेटाबेसमध्ये एन्ट्री करा
    new_contact = TrustedContact(
        user_id=current_user_id,
        name=name,
        phone=phone,
        relation="Self",
        status=sms_status
    )
    
    db.session.add(new_contact)
    db.session.commit()

    return jsonify({
        "name": new_contact.name,
        "phone": new_contact.phone,
        "relation": new_contact.relation,
        "status": new_contact.status
    })


@app.route('/get_emergency_history')
def get_emergency_history():
    # समजा लॉगिन असलेल्या युजरचा ID 1 आहे
    current_user_id = 1 
    
    # डेटाबेस मधून या युजरचे सर्व कॉन्टॅक्ट्स मिळवा
    contacts = TrustedContact.query.filter_by(user_id=current_user_id).all()
    
    history_list = []
    for c in contacts:
        history_list.append({
            "name": c.name,
            "phone": c.phone,
            "status": c.status or "Sent" # जर स्टेटस नसेल तर डिफॉल्ट 'Sent'
        })
    
    return jsonify(history_list)
@app.route("/request_delete_account")
def request_delete_account():
    if 'user_id' in session:
        #user_id = session['user_id']
        # इथे तुम्ही युजरचा स्टेटस 'Pending Delete' करू शकता 
        # किंवा एडमिनला ईमेल पाठवू शकता.
        user = User.query.get(session['user_id'])
        
        if user:
            # २. डेटाबेसमध्ये त्याचे खाते 'Deleted' म्हणून मार्क करा
            user.is_deleted = True 
            db.session.commit()
            
            # ३. युजरचे लॉगिन सेशन पूर्णपणे पुसून टाका (Logout)
            session.clear()
            
            # ४. थेट रजिस्ट्रेशन पेजवर पाठवा (कारण तुम्हाला लॉगिन नको आहे)
            print(f"--- Account Successfully Deleted for: {user.name} ---")
            return redirect(url_for('register')) 
            
    return redirect(url_for('register'))
        # उदाहरणादाखल आपण फक्त फ्लॅश मेसेज दाखवू:
       # session['delete_requested'] = True
        #return redirect(url_for('profile')) 
   #return redirect(url_for('login'))

    

# -------------------------------
# DETAILED REPORT
# -------------------------------
"""@app.route("/report/<int:record_id>")
def detailed_report(record_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # 1. Get the specific record
    record = PredictionRecord.query.get_or_404(record_id)
    
    # 2. Parse the saved JSON
    inputs = json.loads(record.input_json)
    contributions = json.loads(record.explanation_json)
    
    # 3. Fetch History for Trend Chart
    history_records = PredictionRecord.query.filter_by(user_id=session['user_id'])\
        .order_by(PredictionRecord.created_at.asc())\
        .limit(5).all()
    
    dates = [h.created_at.strftime('%m/%d') for h in history_records]
    scores = [round(h.probability * 100, 1) for h in history_records]

    # 4. BMI Calculation Safety Check
    current_bmi = inputs.get('bmi')
    if current_bmi is None:
        h_cm = float(inputs.get('height', 170))
        w_kg = float(inputs.get('weight', 70))
        current_bmi = round(w_kg / ((h_cm / 100) ** 2), 1) if h_cm > 0 else 25.0

    # 5. Parameter Analysis
    parameter_analysis = get_full_parameter_analysis(inputs, current_bmi)
    

    # 6. Radar Chart Normalization
    radar_data = [
        normalize(float(inputs.get('avg_glucose_level', 100)), 50, 200),
        normalize(float(inputs.get('heart_rate', 70)), 40, 140),
        normalize(float(current_bmi), 15, 40),
        normalize(float(inputs.get('spo2', 98)), 80, 100),
        normalize(float(inputs.get('sleep_hours', 7)), 4, 12),
        normalize(float(inputs.get('physical_activity', 1)), 0, 5),
        normalize(float(inputs.get('diet_quality', 2)), 0, 3)
    ]



     # 6. Live Advice Logic (Triggers immediate alerts)
    live_advice = []
    
    # Heart Rate Alert
    hr = float(inputs.get('heart_rate', 0))
    if hr > 100:
        live_advice.append({"type": "danger", "msg": "High Heart Rate Detected", "action": "Please rest for 15 minutes and re-measure."})
    elif hr < 50 and hr > 0:
        live_advice.append({"type": "warning", "msg": "Low Heart Rate Detected", "action": "Consult a professional if you feel dizzy."})

    # Blood Sugar Alert (Based on your 150.0 mg/dL reading)
    glucose = float(inputs.get('avg_glucose_level', 0))
    if glucose > 140:
        live_advice.append({"type": "danger", "msg": "Hyperglycemia (High Blood Sugar)", "action": "Avoid sugary intake immediately and stay hydrated."})

    # SpO2 Alert
    spo2 = float(inputs.get('spo2', 100))
    if spo2 < 92:
        live_advice.append({"type": "danger", "msg": "Low Oxygen Saturation", "action": "Seek medical advice if shortness of breath persists."})
    # 7. Live Advice Logic
    live_advice = []
    if float(inputs.get('heart_rate', 0)) > 100:
        live_advice.append({
            "type": "danger", 
            "msg": "High Heart Rate", 
            "action": "Rest and re-measure in 10 mins."
        })

    return render_template("detailed_report.html", 
                           record=record, 
                           inputs=inputs, 
                           contributions=contributions,
                           live_advice=live_advice,
                           trend_dates=dates,
                           trend_scores=scores,
                           parameter_analysis=parameter_analysis,
                           radar_data=radar_data)"""


# --- ADD THESE HELPERS AT THE TOP (After imports) ---

def normalize(value, min_val, max_val):
    try:
        score = ((float(value) - min_val) / (max_val - min_val)) * 100
        return max(0, min(100, score))
    except:
        return 50

def normalize_inverse(value, min_val, max_val):
    try:
        score = ((float(value) - min_val) / (max_val - min_val)) * 100
        inverted = 100 - score
        return max(0, min(100, inverted))
    except:
        return 50
def get_bmi_score(bmi):
    try:
        bmi = float(bmi)
        if 18.5 <= bmi <= 25:
            return 100  # Perfect score (Edge par dikhega)
        elif bmi < 18.5:
            # Underweight: Jitna kam, utna center ke paas
            return max(0, (bmi / 18.5) * 100)
        else:
            # Overweight: 40+ BMI par score 0 (Center)
            score = 100 - ((bmi - 25) / (40 - 25)) * 100
            return max(0, score)
    except:
        return 50

@app.route('/help')
def help_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_name = session.get('user_name', 'Minal')
    return render_template('help.html', user_name=user_name)

@app.route('/send-emergency', methods=['POST'])
def send_emergency():
    data = request.json
    # Logic to actually send email or SMS goes here
    print(f"REPORT DISPATCHED: ID {data['record_id']} to {data['contact']['phone']}")
    return jsonify({"status": "success"})



# --- NOW YOUR ROUTE WILL WORK ---
"""
@app.route("/report/<int:record_id>")
def detailed_report(record_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    record = PredictionRecord.query.get_or_404(record_id)
    inputs = json.loads(record.input_json)
    contributions = json.loads(record.explanation_json)


    
    # --- 1. TREND DATA ---
    history_records = PredictionRecord.query.filter_by(user_id=session['user_id'])\
        .order_by(PredictionRecord.created_at.asc())\
        .limit(5).all()
    dates = [h.created_at.strftime('%m/%d') for h in history_records]
    scores = [round(h.probability * 100, 1) for h in history_records]

    # --- 2. DYNAMIC ADVICE (Fixing the Heart Rate mismatch) ---
    live_advice = []
    glucose = float(inputs.get('avg_glucose_level', 0))
    hr = float(inputs.get('heart_rate', 0))
    spo2 = float(inputs.get('spo2', 100))

    if glucose > 140:
        live_advice.append({"type": "danger", "msg": "High Blood Sugar", "action": f"Current: {glucose} mg/dL. Limit carbs immediately."})
    
    if hr < 55:
        live_advice.append({"type": "warning", "msg": "Low Heart Rate (Bradycardia)", "action": "Consult a doctor if you feel dizzy or faint."})
    elif hr > 100:
        live_advice.append({"type": "danger", "msg": "High Heart Rate (Tachycardia)", "action": "Avoid caffeine and rest."})

    if spo2 < 94:
        live_advice.append({"type": "danger", "msg": "Low Oxygen", "action": "Check for sleep apnea or respiratory issues."})

    # --- 3. RADAR DATA ---
    current_bmi = inputs.get('bmi', 25.0)
    radar_data = [
        normalize_inverse(glucose, 70, 200),
        normalize(hr, 40, 100),
        get_bmi_score(current_bmi),
        normalize(float(inputs.get('spo2', 98)), 85, 100),
        normalize(float(inputs.get('sleep_hours', 7)), 0, 10),
        normalize(float(inputs.get('physical_activity', 0)), 0, 5),
        85 # Diet
    ]

    return render_template("detailed_report.html", 
                           record=record, 
                           inputs=inputs, 
                           contributions=contributions,
                           live_advice=live_advice, 
                           radar_data=radar_data,
                           trend_dates=dates,
                           trend_scores=scores,
                           parameter_analysis=get_full_parameter_analysis(inputs, current_bmi))
    
    """




@app.route("/report/<int:record_id>")
def detailed_report(record_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    record = PredictionRecord.query.get_or_404(record_id)
    inputs = json.loads(record.input_json)
    contributions = json.loads(record.explanation_json)

    # --- 1. TREND DATA (FIXED LOGIC) ---
    # Sabse pehle LATEST 5 records uthao (Descending order)
    recent_history = PredictionRecord.query.filter_by(user_id=session['user_id'])\
        .order_by(PredictionRecord.created_at.desc())\
        .limit(5).all()
    
    # Ab unhe reverse kar do taaki Graph mein Purana -> Naya (Left to Right) dikhe
    history_records = recent_history[::-1]

    # Dates mein Time (%H:%M) bhi add karo taaki same date ke records alag dikhen
    dates = [h.created_at.strftime('%d/%m %H:%M') for h in history_records]
    scores = [round(h.probability * 100, 1) for h in history_records]

    # --- 2. DYNAMIC ADVICE ---
    live_advice = []
    glucose = float(inputs.get('avg_glucose_level', 0))
    hr = float(inputs.get('heart_rate', 0))
    spo2 = float(inputs.get('spo2', 100))

    if glucose > 140:
        live_advice.append({"type": "danger", "msg": "High Blood Sugar", "action": f"Current: {glucose} mg/dL. Limit carbs immediately."})
    
    if hr < 55:
        live_advice.append({"type": "warning", "msg": "Low Heart Rate (Bradycardia)", "action": "Consult a doctor if you feel dizzy or faint."})
    elif hr > 100:
        live_advice.append({"type": "danger", "msg": "High Heart Rate (Tachycardia)", "action": "Avoid caffeine and rest."})

    if spo2 < 94:
        live_advice.append({"type": "danger", "msg": "Low Oxygen", "action": "Check for sleep apnea or respiratory issues."})

    # --- 3. RADAR DATA ---
    current_bmi = inputs.get('bmi', 25.0)
    radar_data = [
        normalize_inverse(glucose, 70, 200),
        normalize(hr, 40, 100),
        get_bmi_score(current_bmi),
        normalize(float(inputs.get('spo2', 98)), 85, 100),
        normalize(float(inputs.get('sleep_hours', 7)), 0, 10),
        normalize(float(inputs.get('physical_activity', 0)), 0, 5),
        85 # Diet
    ]

    return render_template("detailed_report.html", 
                           record=record, 
                           inputs=inputs, 
                           contributions=contributions,
                           live_advice=live_advice, 
                           radar_data=radar_data,
                           trend_dates=dates,
                           trend_scores=scores,
                           parameter_analysis=get_full_parameter_analysis(inputs, current_bmi))



    
    
    
def get_full_parameter_analysis(inputs, bmi):
    analysis = []
    # BMI ko metrics mein add kiya
    num_metrics = {
        "avg_glucose_level": {"min": 70, "max": 100, "label": "Blood Sugar", "unit": "mg/dL"},
        "heart_rate": {"min": 60, "max": 100, "label": "Resting Heart Rate", "unit": "BPM"},
        "spo2": {"min": 95, "max": 100, "label": "Oxygen Saturation", "unit": "%"},
        "sleep_hours": {"min": 7, "max": 9, "label": "Sleep Duration", "unit": "hrs"},
        "physical_activity": {"min": 0.5, "max": 3, "label": "Physical Activity", "unit": "hrs/day"}
    }

    # 1. BMI Analysis (Special Logic)
    bmi_val = float(bmi)
    if bmi_val < 18.5:
        status, color, sug = "Underweight", "var(--warning)", "Increase calorie intake."
    elif bmi_val > 25:
        status, color, sug = "Overweight", "var(--danger)", "Focus on diet and exercise."
    else:
        status, color, sug = "Normal", "var(--success)", "Perfect BMI maintained."
    
    analysis.append({
        "param": "Body Mass Index (BMI)", 
        "val": f"{bmi_val}", 
        "status": status, 
        "color": color, 
        "sug": sug
    })

    # 2. Baki Metrics ka loop
    for key, bounds in num_metrics.items():
        val = float(inputs.get(key, 0))
        if val < bounds["min"]:
            status, color, sug = "Low", "var(--warning)", f"Increase {bounds['label']}."
        elif val > bounds["max"]:
            status, color, sug = "High", "var(--danger)", f"Reduce {bounds['label']}."
        else:
            status, color, sug = "Optimal", "var(--success)", "Maintained well."
        
        analysis.append({
            "param": bounds['label'], 
            "val": f"{val} {bounds['unit']}", 
            "status": status, 
            "color": color, 
            "sug": sug
        })

    return analysis
    
    """
def get_full_parameter_analysis(inputs, bmi):
    analysis = []
    num_metrics = {
        "avg_glucose_level": {"min": 70, "max": 100, "label": "Blood Sugar", "unit": "mg/dL"},
        "heart_rate": {"min": 60, "max": 100, "label": "Resting Heart Rate", "unit": "BPM"},
        "spo2": {"min": 95, "max": 100, "label": "Oxygen Saturation", "unit": "%"},
        "sleep_hours": {"min": 7, "max": 9, "label": "Sleep Duration", "unit": "hrs"},
        "physical_activity": {"min": 0.5, "max": 3, "label": "Physical Activity", "unit": "hrs/day"}
    }

    for key, bounds in num_metrics.items():
        val = float(inputs.get(key, 0))
        if val < bounds["min"]:
            status, color, sug = "Low", "var(--warning)", f"Increase {bounds['label']}."
        elif val > bounds["max"]:
            status, color, sug = "High", "var(--danger)", f"Reduce {bounds['label']}."
        else:
            status, color, sug = "Optimal", "var(--success)", "Maintained well."
        
        analysis.append({
            "param": bounds['label'], 
            "val": f"{val} {bounds['unit']}", 
            "status": status, 
            "color": color, 
            "sug": sug
        })

    return analysis

"""
@app.route("/show_map")
def show_map():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("map.html")


@app.route('/save_profile', methods=['POST'])
def save_profile():
    # User kadun aleli mahiti
    t_name = request.form.get('name')
    t_phone = request.form.get('phone')
    t_relation = request.form.get('relation')

    # Navin Trusted Contact banva (History sathi)
    new_contact = TrustedContact(
        name=t_name, 
        phone=t_phone, 
        relation=t_relation, 
        user_id=current_user.id # Login aslelya user chi ID
    )

    db.session.add(new_contact)
    db.session.commit()
    return redirect('/dashboard')

@app.route('/send_emergency_sms', methods=['POST'])
def send_emergency_sms():
    contacts = TrustedContact.query.filter_by(user_id=session['user_id']).all()
    message = "EMERGENCY! High Stroke Risk Detected. View: http://neuroscan.com/profile/101"
    
    for contact in contacts:
        # Yithe tumhi Twilio kiwa kontihi SMS API call karu shakta
        print(f"Sending SMS to {contact.phone}") 
        # send_sms_via_api(contact.phone, message)
        
    return jsonify({"status": "success"})    

@app.route('/log_sms', methods=['POST'])
def log_sms():
    print("---------------------------------------")
    print("🚀 ALERT: SMS Protocol Triggered!")
    print("📲 Status: SMS App Opened on User Device")
    print("---------------------------------------")
    return {"status": "success"}


@app.route('/get_vitals')
def get_vitals():
    # Ithe tumcha current live data dya
    # Jar tumhi variable vaprat asal tar: return jsonify(latest_vitals)
    return {
        "heart_rate": 78,  # Ithe real-time value yeil
        "spo2": 97         # Ithe real-time value yeil
    }


@app.route('/get_latest_vitals')
def get_latest_vitals():
    # Database madhun sarvat shevatcha record ghyava
    latest = PredictionRecord.query.order_by(PredictionRecord.created_at.desc()).first()
    
    if latest:
        import json
        vitals = json.loads(latest.input_json)
        return {
            "success": True,
            "heartRate": vitals.get('heartRate', 0),
            "spo2": vitals.get('spo2', 0),
            "steps": vitals.get('steps', 0),
            "sleepHours": vitals.get('sleepHours', 0)
        }
    return {"success": False}



@app.route('/nearby-hospitals')
def nearby_hospitals():
    return render_template('map_hospitals.html') 



@app.route('/calculator')
def alcohol_calculator():
    return render_template('calculator.html') # The advanced ML/ABV code we just built



# Dono options allow karne ke liye do routes dein
@app.route("/")
@app.route("/home")
def home():
    user_name = session.get('user_name', None)
    return render_template("home.html", name=user_name)


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
