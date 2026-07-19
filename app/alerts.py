import smtplib
import os
from email.message import EmailMessage

def send_doctor_report(prediction_result, patient_name, receiver_email, html_content):
    # --- CONFIGURATION ---
    SENDER_EMAIL = "divyanipatil60@gmail.com" 
    APP_PASSWORD = "wbrhsmjlztfsgajb" 

    msg = EmailMessage()
    msg['Subject'] = f"Health Alert: Stroke Risk Analysis Report - {patient_name}"
    msg['From'] = f"NeuroScan AI Support <{SENDER_EMAIL}>"
    msg['To'] = receiver_email

    # Email Body (Text version)
    msg.set_content("A detailed clinical report is attached.")

    # HTML content ko attachment ki tarah add karein
    msg.add_attachment(
        html_content.encode('utf-8'), # String ko bytes mein convert kiya
        maintype='text',
        subtype='html',
        filename=f"Stroke_Risk_Report_{patient_name}.html"
    )

    # --- SENDING ---
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
            return True
    except Exception as e:
        print(f"❌ Email Error: {e}")
        return False