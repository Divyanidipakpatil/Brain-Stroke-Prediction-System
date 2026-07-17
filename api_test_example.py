"""
Example of how to use the Explainability Engine API endpoint
"""

import requests
import json

def test_report_api():
    """Test the /generate_report API endpoint"""
    
    # Example prediction data (same format from requirements)
    prediction_data = {
        "probability": 57.18,
        "category": "High", 
        "contributions": [
            {
                "name": "age",
                "value": 17.36,
                "direction": "increases",
                "user_value": 56
            },
            {
                "name": "hypertension", 
                "value": 14.2,
                "direction": "increases",
                "user_value": "Yes"
            },
            {
                "name": "bmi",
                "value": 10.3,
                "direction": "increases", 
                "user_value": 29.8
            },
            {
                "name": "glucose",
                "value": 9.7,
                "direction": "increases",
                "user_value": 140
            },
            {
                "name": "diet_quality",
                "value": 14.04,
                "direction": "increases",
                "user_value": "Poor"
            }
        ]
    }
    
    # API endpoint (assuming Flask app is running on localhost:5000)
    url = "http://localhost:5000/generate_report"
    
    try:
        # Make POST request to generate report
        response = requests.post(url, json=prediction_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Report generated successfully!")
            print("\n--- GENERATED REPORT ---")
            print(result["report"])
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            
    except requests.exceptions.ConnectionError:
        print("⚠️  Flask app not running. Start the app with:")
        print("   cd app")
        print("   python main.py")
        print("Then run this test again.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_report_api()
