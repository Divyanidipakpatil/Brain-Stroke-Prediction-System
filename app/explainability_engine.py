"""
Explainability & Risk-Guidance Engine for Stroke Prediction Web Application
Generates clear explanations, risk-increasing factors, diet plan, daily routine recommendations, and structured reports.
"""

import json
from typing import Dict, List, Any


class StrokeRiskExplainer:
    """
    Comprehensive engine for explaining stroke risk predictions and providing personalized guidance.
    """
    
    def __init__(self):
        # Risk factor guidance database with explanations and suggestions
        self.RISK_FACTOR_GUIDANCE = {
            "age": {
                "explanation": "As we age, blood vessels naturally become less flexible and more prone to damage. This reduces blood flow to the brain and increases the likelihood of clot formation or vessel rupture.",
                "suggestions": [
                    "Schedule annual health check-ups to monitor blood pressure and heart health",
                    "Engage in regular physical activity (30 minutes daily) to maintain cardiovascular health",
                    "Follow a heart-healthy diet rich in fruits, vegetables, and whole grains",
                    "Stay socially active and mentally engaged to support overall brain health"
                ]
            },
            "hypertension": {
                "explanation": "High blood pressure damages the delicate inner lining of blood vessels over time, making them narrower and less elastic. This increases the risk of both ischemic (clot-based) and hemorrhagic (bleeding) strokes.",
                "suggestions": [
                    "Monitor blood pressure regularly at home and keep a log",
                    "Reduce sodium intake to less than 2,300mg per day (ideally 1,500mg)",
                    "Take prescribed blood pressure medications consistently as directed",
                    "Practice stress reduction techniques like deep breathing or meditation"
                ]
            },
            "heart_disease": {
                "explanation": "Heart conditions like atrial fibrillation can cause blood clots to form in the heart, which can travel to the brain and cause stroke. Coronary artery disease also shares risk factors with stroke.",
                "suggestions": [
                    "Follow prescribed heart medications exactly as directed",
                    "Report any heart palpitations, chest pain, or shortness of breath immediately",
                    "Maintain a healthy weight through balanced diet and regular exercise",
                    "Avoid smoking and limit alcohol consumption significantly"
                ]
            },
            "avg_glucose_level": {
                "explanation": "High glucose levels damage blood vessels throughout the body, including those in the brain. This damage can lead to atherosclerosis (hardening of arteries) and increase clot formation risk.",
                "suggestions": [
                    "Monitor blood glucose levels regularly and maintain a log",
                    "Follow a low-glycemic, high-fiber diet with controlled carbohydrate portions",
                    "Take diabetes medications or insulin exactly as prescribed",
                    "Engage in regular physical activity to improve insulin sensitivity"
                ]
            },
            "bmi": {
                "explanation": "Excess weight increases stroke risk through multiple mechanisms: higher blood pressure, increased cholesterol levels, greater strain on the heart, and inflammation throughout the body.",
                "suggestions": [
                    "Aim for gradual weight loss of 1-2 pounds per week through sustainable changes",
                    "Practice portion control using smaller plates and measuring serving sizes",
                    "Increase physical activity gradually to 150 minutes of moderate exercise weekly",
                    "Focus on nutrient-dense, low-calorie foods like vegetables and lean proteins"
                ]
            },
            "physical_activity": {
                "explanation": "Sedentary lifestyle contributes to poor circulation, weight gain, high blood pressure, and reduced cardiovascular fitness - all major stroke risk factors.",
                "suggestions": [
                    "Start with 10-minute walks and gradually increase duration and intensity",
                    "Find enjoyable activities like swimming, dancing, or cycling",
                    "Take stairs instead of elevators when possible",
                    "Set a daily step goal (start with 5,000 steps and work toward 10,000)"
                ]
            },
            "diet_quality": {
                "explanation": "Poor diet contributes to stroke risk through high sodium (raising blood pressure), unhealthy fats (increasing cholesterol), excess calories (causing obesity), and lack of protective nutrients.",
                "suggestions": [
                    "Adopt the DASH diet or Mediterranean eating pattern",
                    "Limit processed foods, fast food, and restaurant meals",
                    "Eat at least 5 servings of fruits and vegetables daily",
                    "Choose whole grains over refined grains and healthy fats over saturated fats"
                ]
            },
            "sleep_hours": {
                "explanation": "Both insufficient and excessive sleep can disrupt hormone regulation, increase inflammation, raise blood pressure, and impair metabolic function - all increasing stroke risk.",
                "suggestions": [
                    "Aim for 7-8 hours of quality sleep per night",
                    "Maintain consistent sleep and wake times, even on weekends",
                    "Create a relaxing bedtime routine and optimize your sleep environment",
                    "Avoid caffeine, heavy meals, and screens 2-3 hours before bedtime"
                ]
            },
            "alcohol_intake": {
                "explanation": "Excessive alcohol raises blood pressure, contributes to irregular heart rhythms, increases bleeding risk, and can lead to weight gain - all factors that increase stroke risk.",
                "suggestions": [
                    "Limit alcohol to no more than 1 drink per day for women, 2 for men",
                    "Consider alcohol-free days several times per week",
                    "Avoid binge drinking (4+ drinks for women, 5+ for men in one occasion)",
                    "Choose lower-calorie, lower-sugar alcoholic beverages when drinking"
                ]
            },
            "family_stroke_history": {
                "explanation": "Genetic factors can predispose individuals to conditions that increase stroke risk, such as high blood pressure, diabetes, or blood clotting disorders.",
                "suggestions": [
                    "Be extra vigilant about controlling modifiable risk factors",
                    "Inform healthcare providers about family history for personalized screening",
                    "Consider genetic counseling if multiple family members had strokes at young ages",
                    "Maintain detailed health records and share them with all healthcare providers"
                ]
            },
            "smoking_status": {
                "explanation": "Smoking damages blood vessels, increases blood pressure and heart rate, makes blood more likely to clot, and reduces oxygen in the blood - dramatically increasing stroke risk.",
                "suggestions": [
                    "Quit smoking completely - every smoke-free day reduces risk",
                    "Use nicotine replacement therapy or medications as prescribed by doctor",
                    "Join a smoking cessation program or use quit-smoking apps",
                    "Avoid secondhand smoke exposure whenever possible"
                ]
            }
        }
        
        # Diet recommendations for stroke prevention
        self.DIET_RECOMMENDATIONS = {
            "foods_to_include": [
                "Leafy greens (spinach, kale, collard greens) - rich in vitamins and minerals",
                "Berries (blueberries, strawberries) - high in antioxidants",
                "Whole grains (oats, brown rice, quinoa) - fiber and B vitamins",
                "Fatty fish (salmon, mackerel, sardines) - omega-3 fatty acids",
                "Nuts and seeds (walnuts, flaxseeds, chia seeds) - healthy fats",
                "Legumes (beans, lentils, chickpeas) - protein and fiber",
                "Low-fat dairy (yogurt, milk) - calcium and vitamin D",
                "Olive oil - monounsaturated fats and antioxidants",
                "Garlic and onions - blood pressure benefits",
                "Avocados - potassium and healthy fats"
            ],
            "foods_to_avoid": [
                "Processed meats (bacon, sausage, deli meats) - high sodium and preservatives",
                "Sugary drinks (soda, sweetened juices) - empty calories and inflammation",
                "Refined grains (white bread, pasta, white rice) - low nutritional value",
                "Fried foods - unhealthy fats and calories",
                "High-sodium foods (canned soups, frozen dinners) - blood pressure impact",
                "Excessive red meat - saturated fat content",
                "Full-fat dairy products - saturated fats",
                "Commercial baked goods - trans fats and refined sugars",
                "Alcohol (excessive) - blood pressure and bleeding risk",
                "Artificial sweeteners - may affect glucose metabolism"
            ]
        }
        
        # Lifestyle recommendations
        self.LIFESTYLE_RECOMMENDATIONS = {
            "exercise": [
                "Walking: Start with 15-20 minutes daily, gradually increase to 30-45 minutes",
                "Swimming or water aerobics: Low-impact, full-body workout",
                "Cycling: Stationary or outdoor, great for cardiovascular health",
                "Strength training: 2-3 times per week with light weights or resistance bands",
                "Yoga or tai chi: Improves balance, flexibility, and reduces stress",
                "Dancing: Fun way to get cardio exercise"
            ],
            "stress_management": [
                "Deep breathing exercises: 5-10 minutes, 2-3 times daily",
                "Meditation or mindfulness: Use apps like Calm or Headspace for guidance",
                "Progressive muscle relaxation: Tense and relax muscle groups systematically",
                "Journaling: Write down thoughts and worries to process emotions",
                "Nature walks: Spend time in green spaces to reduce stress",
                "Social connection: Maintain relationships with friends and family"
            ],
            "sleep_routine": [
                "Consistent schedule: Same bedtime and wake time daily",
                "Dark, cool room: Keep temperature between 60-67°F (15-19°C)",
                "No screens: Remove phones, tablets, TVs from bedroom",
                "Relaxing routine: Reading, gentle stretching, or warm bath before bed",
                "Limit fluids: Reduce liquid intake 2 hours before bedtime",
                "Avoid caffeine: No coffee, tea, or caffeinated soda after 2 PM"
            ],
            "habits_to_build": [
                "Daily blood pressure monitoring",
                "Regular medication adherence",
                "Weekly meal planning and preparation",
                "Daily physical activity (even short walks count)",
                "Regular medical check-ups and screenings",
                "Staying hydrated with water throughout the day",
                "Practicing gratitude or positive thinking daily"
            ],
            "habits_to_avoid": [
                "Smoking and tobacco use",
                "Excessive alcohol consumption",
                "Sedentary behavior (sitting for long periods)",
                "Skipping medications or medical appointments",
                "Ignoring warning signs or symptoms",
                "Stress eating or emotional eating",
                "Late-night eating or heavy meals before bed"
            ]
        }
    
    def generate_comprehensive_report(self, prediction_data: Dict[str, Any]) -> str:
        """
        Generate a complete stroke risk explanation report based on prediction data.
        
        Args:
            prediction_data: Dictionary containing probability, category, and contributions
            
        Returns:
            Formatted report string with all sections
        """
        probability = prediction_data.get("probability", 0)
        category = prediction_data.get("category", "Unknown")
        contributions = prediction_data.get("contributions", [])
        
        # Filter for risk-increasing factors only
        increasing_factors = [
            contrib for contrib in contributions 
            if contrib.get("direction") == "increases"
        ]
        
        # Generate report sections
        report = self._build_report_structure(
            probability, category, increasing_factors
        )
        
        return report
    
    def _build_report_structure(self, probability: float, category: str, 
                                increasing_factors: List[Dict]) -> str:
        """
        Build the complete report structure following the exact format requirements.
        """
        report_lines = []
        
        # Header
        report_lines.append("Stroke Risk Explanation Report")
        report_lines.append("")
        
        # Section 1: Summary of Major Risk-Increasing Factors
        report_lines.append("1. Summary of Major Risk-Increasing Factors")
        summary_line = self._format_summary_line(increasing_factors)
        report_lines.append(summary_line)
        report_lines.append("")
        
        # Section 2: Detailed Explanations
        report_lines.append("2. Detailed Explanations")
        for factor in increasing_factors:
            factor_section = self._generate_factor_explanation(factor)
            report_lines.extend(factor_section)
            report_lines.append("")
        
        # Section 3: Personalized Diet Plan
        report_lines.append("3. Personalized Diet Plan")
        diet_section = self._generate_diet_plan(increasing_factors)
        report_lines.extend(diet_section)
        report_lines.append("")
        
        # Section 4: Daily Lifestyle Routine
        report_lines.append("4. Daily Lifestyle Routine")
        lifestyle_section = self._generate_lifestyle_plan(increasing_factors)
        report_lines.extend(lifestyle_section)
        report_lines.append("")
        
        # Section 5: Final Advice
        report_lines.append("5. Final Advice")
        final_advice = self._generate_final_advice(probability, category)
        report_lines.append(final_advice)
        
        return "\n".join(report_lines)
    
    def _format_summary_line(self, increasing_factors: List[Dict]) -> str:
        """
        Format the summary line of risk factors with their contributions.
        """
        if not increasing_factors:
            return "No significant risk-increasing factors identified."
        
        factor_strings = []
        for factor in increasing_factors:
            name = factor.get("name", "Unknown").replace("_", " ").title()
            user_value = factor.get("user_value", "N/A")
            contribution = factor.get("value", 0)
            
            # Format user value appropriately
            if isinstance(user_value, (int, float)):
                if user_value == int(user_value):
                    user_value = int(user_value)
                formatted_value = str(user_value)
            else:
                formatted_value = str(user_value)
            
            factor_strings.append(f"{name} ({formatted_value}) — {contribution}%")
        
        return "\n".join(factor_strings)
    
    def _generate_factor_explanation(self, factor: Dict) -> List[str]:
        """
        Generate detailed explanation for a single risk factor.
        """
        name = factor.get("name", "unknown")
        value = factor.get("value", 0)
        user_value = factor.get("user_value", "N/A")
        
        # Format display name
        display_name = name.replace("_", " ").title()
        
        # Get guidance data
        guidance = self.RISK_FACTOR_GUIDANCE.get(name, {})
        explanation = guidance.get("explanation", f"{display_name} contributes to increased stroke risk.")
        suggestions = guidance.get("suggestions", [])
        
        # Format user value
        if isinstance(user_value, (int, float)):
            if user_value == int(user_value):
                user_value = int(user_value)
            formatted_value = str(user_value)
        else:
            formatted_value = str(user_value)
        
        lines = []
        lines.append(f"{display_name} ({formatted_value}) — {value}% (Increases Risk)")
        lines.append("Why this increases risk:")
        lines.append(explanation)
        lines.append("How to improve:")
        
        for i, suggestion in enumerate(suggestions, 1):
            lines.append(f"suggestion {i}")
            lines.append(suggestion)
        
        return lines
    
    def _generate_diet_plan(self, increasing_factors: List[Dict]) -> List[str]:
        """
        Generate personalized diet plan based on risk factors.
        """
        lines = []
        
        # Foods to Include
        lines.append("Foods to Include:")
        for food in self.DIET_RECOMMENDATIONS["foods_to_include"]:
            lines.append(f"- {food}")
        lines.append("")
        
        # Foods to Avoid
        lines.append("Foods to Avoid:")
        for food in self.DIET_RECOMMENDATIONS["foods_to_avoid"]:
            lines.append(f"- {food}")
        lines.append("")
        
        # Daily Meal Plan
        lines.append("Daily Meal Plan:")
        lines.append("Breakfast:")
        lines.extend([
            "- Oatmeal with berries and walnuts",
            "- Greek yogurt with fruit and a sprinkle of flaxseeds",
            "- Whole grain toast with avocado and a side of fruit"
        ])
        lines.append("")
        
        lines.append("Lunch:")
        lines.extend([
            "- Large salad with mixed greens, vegetables, grilled chicken or fish",
            "- Quinoa bowl with roasted vegetables and beans",
            "- Whole grain sandwich with lean protein and plenty of vegetables"
        ])
        lines.append("")
        
        lines.append("Dinner:")
        lines.extend([
            "- Baked salmon with roasted vegetables and brown rice",
            "- Vegetable stir-fry with tofu and whole grain noodles",
            "- Grilled chicken or fish with steamed vegetables and sweet potato"
        ])
        lines.append("")
        
        lines.append("Snacks:")
        lines.extend([
            "- Fresh fruit (apple, banana, berries)",
            "- Handful of nuts (almonds, walnuts)",
            "- Vegetable sticks with hummus",
            "- Greek yogurt or cottage cheese"
        ])
        lines.append("")
        
        lines.append("Hydration Guidelines:")
        lines.extend([
            "- Drink 8-10 glasses of water daily",
            "- Limit caffeine to 1-2 cups per day",
            "- Avoid sugary drinks and excessive fruit juices",
            "- Consider herbal teas as alternatives to water"
        ])
        
        return lines
    
    def _generate_lifestyle_plan(self, increasing_factors: List[Dict]) -> List[str]:
        """
        Generate personalized lifestyle improvement plan.
        """
        lines = []
        
        # Exercise
        lines.append("Exercise:")
        for exercise in self.LIFESTYLE_RECOMMENDATIONS["exercise"]:
            lines.append(f"- {exercise}")
        lines.append("")
        
        # Stress Management
        lines.append("Stress Management:")
        for stress_mgmt in self.LIFESTYLE_RECOMMENDATIONS["stress_management"]:
            lines.append(f"- {stress_mgmt}")
        lines.append("")
        
        # Sleep Routine
        lines.append("Sleep Routine:")
        for sleep in self.LIFESTYLE_RECOMMENDATIONS["sleep_routine"]:
            lines.append(f"- {sleep}")
        lines.append("")
        
        # Habits to Build
        lines.append("Habits to Build:")
        for habit in self.LIFESTYLE_RECOMMENDATIONS["habits_to_build"]:
            lines.append(f"- {habit}")
        
        return lines
    
    def _generate_final_advice(self, probability: float, category: str) -> str:
        """
        Generate motivational final advice based on risk level.
        """
        if category == "Low":
            advice_lines = [
                "Your stroke risk is currently low, which is excellent!",
                "Continue maintaining your healthy lifestyle habits.",
                "Regular check-ups and preventive care are still important.",
                "Share your healthy habits with family and friends."
            ]
        elif category == "Medium":
            advice_lines = [
                "Your stroke risk is moderate, but there's room for improvement.",
                "Focus on the specific areas highlighted in this report.",
                "Small, consistent changes can make a big difference over time.",
                "You have the power to reduce your risk significantly."
            ]
        else:  # High
            advice_lines = [
                "Your stroke risk is high, but taking action now can make a real difference.",
                "Start with one or two key changes and build from there.",
                "Work closely with your healthcare provider on a comprehensive plan.",
                "Every positive step you take moves you toward better health."
            ]
        
        return "\n".join(advice_lines)
    
    def get_risk_factor_guidance(self, factor_name: str) -> Dict[str, Any]:
        """
        Get guidance information for a specific risk factor.
        """
        return self.RISK_FACTOR_GUIDANCE.get(factor_name, {})
    
    def get_diet_recommendations(self) -> Dict[str, List[str]]:
        """
        Get general diet recommendations for stroke prevention.
        """
        return self.DIET_RECOMMENDATIONS
    
    def get_lifestyle_recommendations(self) -> Dict[str, List[str]]:
        """
        Get lifestyle recommendations for stroke prevention.
        """
        return self.LIFESTYLE_RECOMMENDATIONS


# Global instance for easy import
explainer = StrokeRiskExplainer()

# Export the guidance dictionary for use in other modules
RISK_FACTOR_GUIDANCE = explainer.RISK_FACTOR_GUIDANCE
