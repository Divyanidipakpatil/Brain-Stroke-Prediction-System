# # app/models.py

# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime

# db = SQLAlchemy()

# # -----------------------
# # ✅ User Model
# # -----------------------


# # class User(db.Model):
# #     __tablename__ = 'user'

# #     id = db.Column(db.Integer, primary_key=True)
# #     name = db.Column(db.String(100), nullable=False)
# #     email = db.Column(db.String(120), unique=True, nullable=False)
# #     password = db.Column(db.String(100), nullable=False)
# #     is_deleted = db.Column(db.Boolean, default=False)
    
# #     # Relationship to other tables
# #     predictions = db.relationship("PredictionRecord", backref="user", lazy=True)
# #     profiles = db.relationship("UserProfile", backref="user", lazy=True)
# #     work_info = db.relationship("UserWorkInfo", backref="user", lazy=True)

# #     contacts = db.relationship("TrustedContact", backref="user", lazy=True)

# #     def __repr__(self):
# #         return f"<User {self.name}>"

# # # -----------------------
# # # ✅ User Profile Model
# # # Stores the data used for the ML Model Prediction
# # # -----------------------
# # class UserProfile(db.Model):
# #     __tablename__ = 'user_profile'

# #     id = db.Column(db.Integer, primary_key=True)
# #     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# #     # Demographic attributes
# #     age = db.Column(db.Float, nullable=False)
# #     gender = db.Column(db.String(20))
# #     ever_married = db.Column(db.String(10))
# #     Residence_type = db.Column(db.String(20))
# #     smoking_status = db.Column(db.String(30))
    
# #     # Medical history (0/1)
# #     hypertension = db.Column(db.Integer, default=0)
# #     heart_disease = db.Column(db.Integer, default=0)
# #     family_stroke_history = db.Column(db.Integer, default=0)
    
# #     # Vital signs / Wearable data
# #     avg_glucose_level = db.Column(db.Float)
# #     bmi = db.Column(db.Float)
# #     heart_rate = db.Column(db.Float)
# #     spo2 = db.Column(db.Float)
# #     steps = db.Column(db.Integer)
    
# #     # Lifestyle factors
# #     physical_activity = db.Column(db.Float) # Score based on your dataset logic
# #     sleep_hours = db.Column(db.Float)
# #     alcohol_intake = db.Column(db.Float)
# #     diet_quality = db.Column(db.Integer) # Categorical/Ordinal scale

# #     created_at = db.Column(db.DateTime, default=datetime.utcnow)

# #     def __repr__(self):
# #         return f"<UserProfile {self.id} - User {self.user_id}>"



# # -----------------------
# # ✅ User Profile Model
# # -----------------------
# class UserProfile(db.Model):
#     __tablename__ = 'user_profile'

#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

#     # --- ADDRESS COLUMN ADD KIYA HAI ---
#     address = db.Column(db.String(255), nullable=True) 

#     # Demographic attributes
#     age = db.Column(db.Float, nullable=False, default=20.0)
#     gender = db.Column(db.String(20))
#     ever_married = db.Column(db.String(10))
#     Residence_type = db.Column(db.String(20))
#     smoking_status = db.Column(db.String(30))
    
#     # Medical history
#     hypertension = db.Column(db.Integer, default=0)
#     heart_disease = db.Column(db.Integer, default=0)
#     family_stroke_history = db.Column(db.Integer, default=0)
    
#     # Vital signs / Wearable data
#     avg_glucose_level = db.Column(db.Float)
#     bmi = db.Column(db.Float)
#     heart_rate = db.Column(db.Float)
#     spo2 = db.Column(db.Float)
#     steps = db.Column(db.Integer)
    
#     # Lifestyle factors
#     physical_activity = db.Column(db.Float)
#     sleep_hours = db.Column(db.Float)
#     alcohol_intake = db.Column(db.Float)
#     diet_quality = db.Column(db.Integer)

#     created_at = db.Column(db.DateTime, default=datetime.utcnow)

#     def __repr__(self):
#         return f"<UserProfile {self.id} - User {self.user_id}>"


# # -----------------------
# # ✅ Prediction Record Model
# # -----------------------
# class PredictionRecord(db.Model):
#     __tablename__ = 'prediction_record'

#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    
#     # Store the exact input sent to the model for audit trails
#     input_json = db.Column(db.Text, nullable=False) 
#     probability = db.Column(db.Float, nullable=False)
#     category = db.Column(db.String(50), nullable=False) # e.g., "High Risk"
    
#     # SHAP explanations/Feature importance
#     explanation_json = db.Column(db.Text) 
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)

#     def __repr__(self):
#         return f"<PredictionRecord {self.id} - User {self.user_id}>"

# # -----------------------
# # ✅ User Work Info Model
# # -----------------------
# class UserWorkInfo(db.Model):
#     __tablename__ = 'user_work_info'

#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

#     work_type = db.Column(db.String(50))
#     work_hours = db.Column(db.Float, nullable=True)
#     work_stress_level = db.Column(db.Integer, nullable=True)
#     shift_work = db.Column(db.Integer, nullable=True)

#     created_at = db.Column(db.DateTime, default=datetime.utcnow)

#     def __repr__(self):
#         return f"<UserWorkInfo {self.id} - User {self.user_id}>"

# # -----------------------
# # ✅ Trusted Contacts Model
# # Stores emergency contacts for each user
# # -----------------------
# class TrustedContact(db.Model):
#     __tablename__ = 'trusted_contact'

#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

#     name = db.Column(db.String(100), nullable=False)
#     phone = db.Column(db.String(20), nullable=False)
#     relation = db.Column(db.String(50), default="Self")
#     # Ha column add kelya mule konta contact kadhi save kela te kalel (History sathi)
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
#     # SMS status: 'Sent', 'Failed', 'Verified'
#     status = db.Column(db.String(20), default="Pending") 
    
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)

#     def __repr__(self):
#         return f"<TrustedContact {self.name} - User {self.user_id}>"







# app/models.py

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# -----------------------
# ✅ User Model (IMPORTANT: Comments hata diye hain)
# -----------------------
class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    # Phone column yahan add kiya hai profile update ke liye
    phone = db.Column(db.String(20), nullable=True) 
    is_deleted = db.Column(db.Boolean, default=False)
    
    # Relationships
    predictions = db.relationship("PredictionRecord", backref="user", lazy=True)
    profiles = db.relationship("UserProfile", backref="user", lazy=True)
    work_info = db.relationship("UserWorkInfo", backref="user", lazy=True)
    contacts = db.relationship("TrustedContact", backref="user", lazy=True)

    def __repr__(self):
        return f"<User {self.name}>"

# -----------------------
# ✅ User Profile Model
# -----------------------
class UserProfile(db.Model):
    __tablename__ = 'user_profile'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # Address column added
    address = db.Column(db.String(255), nullable=True) 

    # Demographic attributes
    age = db.Column(db.Float, nullable=False, default=20.0)
    gender = db.Column(db.String(20))
    ever_married = db.Column(db.String(10))
    Residence_type = db.Column(db.String(20))
    smoking_status = db.Column(db.String(30))
    
    # Medical history
    hypertension = db.Column(db.Integer, default=0)
    heart_disease = db.Column(db.Integer, default=0)
    family_stroke_history = db.Column(db.Integer, default=0)
    
    # Vital signs / Wearable data
    avg_glucose_level = db.Column(db.Float)
    bmi = db.Column(db.Float)
    heart_rate = db.Column(db.Float)
    spo2 = db.Column(db.Float)
    steps = db.Column(db.Integer)
    
    # Lifestyle factors
    physical_activity = db.Column(db.Float)
    sleep_hours = db.Column(db.Float)
    alcohol_intake = db.Column(db.Float)
    diet_quality = db.Column(db.Integer)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<UserProfile {self.id} - User {self.user_id}>"

# -----------------------
# ✅ Prediction Record Model
# -----------------------
class PredictionRecord(db.Model):
    __tablename__ = 'prediction_record'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    input_json = db.Column(db.Text, nullable=False) 
    probability = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    explanation_json = db.Column(db.Text) 
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# -----------------------
# ✅ User Work Info Model
# -----------------------
class UserWorkInfo(db.Model):
    __tablename__ = 'user_work_info'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    work_type = db.Column(db.String(50))
    work_hours = db.Column(db.Float, nullable=True)
    work_stress_level = db.Column(db.Integer, nullable=True)
    shift_work = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# -----------------------
# ✅ Trusted Contacts Model
# -----------------------
class TrustedContact(db.Model):
    __tablename__ = 'trusted_contact'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    relation = db.Column(db.String(50), default="Self")
    status = db.Column(db.String(20), default="Pending") 
    created_at = db.Column(db.DateTime, default=datetime.utcnow)