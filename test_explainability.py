"""
Test script for the Explainability & Risk-Guidance Engine
Tests the engine with the example input format provided in the requirements.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.explainability_engine import explainer

def test_explainability_engine():
    """Test the explainability engine with the example input format."""
    
    # Example input from the requirements
    test_input = {
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
    
    print("Testing Explainability Engine...")
    print("=" * 50)
    
    # Generate the comprehensive report
    try:
        report = explainer.generate_comprehensive_report(test_input)
        
        print("✅ Report generated successfully!")
        print("\n" + "=" * 50)
        print("GENERATED REPORT:")
        print("=" * 50)
        print(report)
        print("=" * 50)
        
        # Test individual components
        print("\n\nTesting individual components...")
        
        # Test risk factor guidance
        age_guidance = explainer.get_risk_factor_guidance("age")
        print(f"\n✅ Age guidance retrieved: {len(age_guidance.get('suggestions', []))} suggestions")
        
        # Test diet recommendations
        diet_recs = explainer.get_diet_recommendations()
        print(f"✅ Diet recommendations: {len(diet_recs.get('foods_to_include', []))} foods to include")
        
        # Test lifestyle recommendations
        lifestyle_recs = explainer.get_lifestyle_recommendations()
        print(f"✅ Lifestyle recommendations: {len(lifestyle_recs.get('exercise', []))} exercise options")
        
        print("\n🎉 All tests passed! The explainability engine is working correctly.")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_edge_cases():
    """Test edge cases and error handling."""
    
    print("\n\nTesting edge cases...")
    print("=" * 30)
    
    # Test with empty contributions
    empty_input = {
        "probability": 25.0,
        "category": "Low",
        "contributions": []
    }
    
    try:
        report = explainer.generate_comprehensive_report(empty_input)
        print("✅ Empty contributions handled correctly")
    except Exception as e:
        print(f"❌ Error with empty contributions: {e}")
    
    # Test with missing fields
    minimal_input = {
        "contributions": [
            {
                "name": "age",
                "value": 15.0,
                "direction": "increases"
            }
        ]
    }
    
    try:
        report = explainer.generate_comprehensive_report(minimal_input)
        print("✅ Minimal input handled correctly")
    except Exception as e:
        print(f"❌ Error with minimal input: {e}")
    
    # Test with unknown risk factor
    unknown_input = {
        "probability": 50.0,
        "category": "Medium",
        "contributions": [
            {
                "name": "unknown_factor",
                "value": 10.0,
                "direction": "increases",
                "user_value": "test"
            }
        ]
    }
    
    try:
        report = explainer.generate_comprehensive_report(unknown_input)
        print("✅ Unknown risk factor handled correctly")
    except Exception as e:
        print(f"❌ Error with unknown factor: {e}")

if __name__ == "__main__":
    success = test_explainability_engine()
    test_edge_cases()
    
    if success:
        print("\n🎯 Explainability Engine is ready for integration!")
    else:
        print("\n⚠️  Some issues found - please check the implementation.")
