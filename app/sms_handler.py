from twilio.rest import Client

# Twilio Credentials
ACCOUNT_SID = "ACe3fb953cee77b58af62dfd4c8e67dd28" 
AUTH_TOKEN = "15c787f46a1f31d8d8b112bb4cbabe0e"    
TWILIO_NUMBER = "+16812361377"                  
VERIFIED_NUMBER = "+918459881450"               

try:
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
except Exception as e:
    print(f"Twilio Client Error: {e}")
    client = None

def send_stroke_alert(probability, category, phone_number=None):
    try:
        # Agar phone_number nahi milta toh verified number use karein
        target_number = phone_number if phone_number else VERIFIED_NUMBER
        
        if not client:
            print("❌ Twilio client initialize nahi hua.")
            return False

         
        # Formal SMS Body
        percent = round(probability * 100, 2)
        message_body = (
        f"HEALTH ALERT: NeuroScan Stroke Analysis\n"
        f"Patient: Clinical Assessment Completed\n"
        f"Result: {category.upper()} RISK ({percent}%)\n"
        f"Action: Detailed report has been sent to the registered email. "
        f"Please review immediately."
    )

        message = client.messages.create(
            body=message_body,
            from_=TWILIO_NUMBER,
            to=target_number
        )
        
        print(f"🚀 SMS Sent Successfully! SID: {message.sid}")
        return True

    except Exception as e:
        print(f"❌ Twilio SMS Error: {e}")
        return False