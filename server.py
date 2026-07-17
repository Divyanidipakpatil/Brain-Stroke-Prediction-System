from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# This is crucial: it allows your website to pull data from this server
CORS(app) 

# Initial data storage
latest_health_data = {
    "steps": 0,
    "heart_rate": 0,
    "spo2": 0,
    "sleep": 0
}

# 1. This is what your MOBILE APP calls to send data
@app.route('/update-health', methods=['POST'])
def update_health():
    global latest_health_data
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400
        
    # LOGIC: Grab the exact keys the phone is sending
    # In your server.py update_health function:
    latest_health_data = {
    "steps": data.get('steps', 0),
    "heart_rate": data.get('heartRate', 0), # Maps 'heartRate' (phone) to 'heart_rate' (web)
    "spo2": data.get('spo2', 0),
    "sleep": data.get('sleepHours', 0)      # Maps 'sleepHours' to 'sleep'
}
    
    print(f"--- [SERVER UPDATED] HR: {latest_health_data['heart_rate']} ---")
    return jsonify({"status": "success"}), 200

# 2. This is what your WEBSITE calls to show data on the dashboard
@app.route('/get-web-sync', methods=['GET'])
def get_web_sync():
    global latest_health_data
    # This sends the data to your browser
    return jsonify(latest_health_data)

if __name__ == '__main__':
    # host='0.0.0.0' makes it visible to your phone on the same Wi-Fi
    app.run(host='0.0.0.0', port=5001, debug=True)