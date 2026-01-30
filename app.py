from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import pickle
from google import genai
import os

app = Flask(__name__)

# Load model and encoders
with open('oa_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
le_drug = data['le_drug_type']
le_drug_response = data['le_drug_response']
le_comorbidity = data['le_comorbidity']

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)


def calculate_bmi_category(bmi, population):
    """Calculate BMI category based on population and BMI value"""
    if population == 'Asian':
        if bmi < 23:
            return 0
        elif bmi < 25:
            return 1
        else:
            return 2
    else:  # Western
        if bmi < 25:
            return 0
        elif bmi < 30:
            return 1
        else:
            return 2


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.json
        
        age = int(data['age'])
        population = data['population']
        bmi = float(data['bmi'])
        crp = float(data['crp'])
        oa_severity = int(data['oa_severity'])
        smoking = data['smoking']
        pain_score = int(data['pain_score'])
        drug_type = data['drug_type']
        prediction_type = data['prediction_type']
        
        # Calculate BMI category
        bmi_cat = calculate_bmi_category(bmi, population)
        
        # Encode smoking
        smoking_bin = 0 if smoking == 'No' else 1
        
        # Encode drug type
        drug_type_enc = le_drug.transform([drug_type])[0]
        
        # Prepare input DataFrame
        X_input = pd.DataFrame(
            [[age, bmi_cat, crp, oa_severity, smoking_bin, pain_score, drug_type_enc]],
            columns=['Age', 'BMI_Category', 'CRP', 'OA_Severity', 'Smoking', 'Pain_Score', 'Drug_Type_Enc']
        )
        
        # Make prediction
        prediction = model.predict(X_input)[0]
        
        if prediction_type == 'drug_response':
            pred_value = prediction[0]
            response_text = le_drug_response.inverse_transform([pred_value])[0]
            return jsonify({
                'success': True,
                'prediction_type': 'Drug Response',
                'result': response_text
            })
        else:  # comorbidity
            pred_value = prediction[1]
            cluster_text = le_comorbidity.inverse_transform([pred_value])[0]
            return jsonify({
                'success': True,
                'prediction_type': 'Comorbidity Cluster',
                'result': cluster_text
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    try:
        # Check if API key is configured
        if not GEMINI_API_KEY or GEMINI_API_KEY == 'your-api-key-here':
            return jsonify({
                'success': False,
                'error': 'Gemini API key is not configured. Please set GEMINI_API_KEY in app.py or as an environment variable.'
            }), 400
        
        # Get form data
        data = request.json
        
        age = int(data['age'])
        population = data['population']
        bmi = float(data['bmi'])
        crp = float(data['crp'])
        oa_severity = int(data['oa_severity'])
        smoking = data['smoking']
        pain_score = int(data['pain_score'])
        drug_type = data['drug_type']
        pain_locations = data.get('pain_locations', [])
        
        # Format pain locations for the prompt
        pain_info = ""
        if pain_locations:
            pain_by_intensity = {'low': [], 'medium': [], 'high': []}
            for loc in pain_locations:
                pain_by_intensity[loc['intensity']].append(loc['part'])
            
            pain_info = "\n\nSpecific Pain Locations:\n"
            if pain_by_intensity['high']:
                pain_info += f"- SEVERE Pain: {', '.join(pain_by_intensity['high'])}\n"
            if pain_by_intensity['medium']:
                pain_info += f"- MODERATE Pain: {', '.join(pain_by_intensity['medium'])}\n"
            if pain_by_intensity['low']:
                pain_info += f"- MILD Pain: {', '.join(pain_by_intensity['low'])}\n"
        else:
            pain_info = "\n\nNote: No specific pain locations were indicated by the patient.\n"
        
        # Create detailed prompt for Gemini
        prompt = f"""
You are an expert clinical nutritionist and physiotherapist specializing in knee osteoarthritis (OA) management. 

Create a comprehensive, personalized diet and fitness plan for a patient with the following profile:

Patient Demographics:
- Age: {age} years
- Population: {population}
- BMI: {bmi} kg/m²
- Current Smoking Status: {smoking}

Clinical Parameters:
- OA Severity: {oa_severity}/4 (0=mild, 4=severe)
- Overall Pain Score: {pain_score}/10
- CRP Level: {crp} mg/L (inflammatory marker)
- Current Medication: {drug_type}

{pain_info}

CRITICAL INSTRUCTIONS FOR EXERCISE RECOMMENDATIONS:
Based on the specific pain locations indicated above, you MUST:
1. Completely AVOID exercises that directly stress or load painful joints
2. Focus on ALTERNATIVE exercises that work around the painful areas
3. Provide MODIFIED versions of standard exercises for affected areas
4. Suggest SUBSTITUTION exercises when primary movements are contraindicated

For example:
- If knees have severe pain: NO squats, lunges, or jumping; instead recommend pool exercises, seated resistance, and upper body work
- If shoulders have pain: NO overhead pressing; instead recommend modified lateral raises, front raises at lower angles
- If back has pain: NO bending/twisting exercises; instead recommend core bracing, bird dogs, dead bugs
- If hips have pain: NO deep hip flexion exercises; instead recommend supine leg exercises, resistance band work

Instructions:
Generate a detailed, actionable plan with the following sections:

1. Diet Plan Overview
   - Brief summary of dietary goals tailored to reduce inflammation and support joint health
   - Consider CRP level of {crp} mg/L for inflammation management

2. Daily Meal Plan
   Provide specific meals for:
   - Breakfast (with anti-inflammatory ingredients)
   - Mid-morning Snack
   - Lunch (protein-rich, joint-supporting)
   - Evening Snack
   - Dinner (light, nutrient-dense)
   
   For each meal:
   - List specific food items with approximate portions
   - Include rationale (anti-inflammatory properties, joint health, weight management)
   - Consider {population} dietary preferences

3. Key Nutritional Guidelines
   - Foods to emphasize (omega-3 fatty acids, antioxidants, vitamin D, calcium)
   - Foods to limit or completely avoid
   - Daily hydration goals (specific volume)
   - Supplement recommendations with dosages (Vitamin D, Omega-3, Glucosamine, Chondroitin)
   - Timing of meals relative to medication ({drug_type})

4. Exercise & Fitness Plan
   Create a weekly exercise schedule that SPECIFICALLY ADDRESSES the pain locations:
   
   For each painful area, provide:
   - What exercises to COMPLETELY AVOID
   - Alternative exercises that don't stress that area
   - Modified versions of standard exercises
   - Proper form cues to prevent further injury
   
   Weekly Structure:
   - Monday: [Specific exercises avoiding painful areas]
   - Tuesday: [Different muscle groups, avoiding contraindicated movements]
   - Wednesday: [Active recovery or gentle mobility]
   - Thursday: [Strength work around limitations]
   - Friday: [Cardio that doesn't aggravate pain sites]
   - Saturday: [Flexibility and ROM work]
   - Sunday: [Complete rest or very gentle movement]
   
   Include:
   - Duration and repetitions for each exercise
   - Pain management strategies during exercise
   - When to stop and seek medical help
   - Progressive overload strategy as pain improves

5. Pain-Specific Modifications
   For EACH painful body part mentioned, provide:
   - Specific exercises to avoid
   - Alternative exercises
   - Pain relief strategies (ice, heat, compression)
   - Range of motion exercises
   - When to progress to more challenging movements

6. Lifestyle Recommendations
   - Weight management strategies (if BMI indicates need: current {bmi})
   - Daily activity modifications based on pain locations
   - Sleep position recommendations for painful areas
   - Stress management techniques
   - Smoking cessation plan (if applicable: {smoking})

7. Important Safety Considerations
   - Red flags requiring immediate medical attention
   - How to distinguish between "good pain" and "bad pain"
   - Medication timing with meals and exercise
   - Drug-nutrient interactions with {drug_type}
   - When to adjust or stop the plan

FORMATTING REQUIREMENTS:
- Use proper HTML formatting throughout
- Use <h2> for main sections, <h3> for subsections
- Use <p> for paragraphs, <ul> and <li> for lists
- Use <strong> for emphasis on important points
- DO NOT use markdown symbols (**, __, *, etc.)
- Make exercise contraindications very clear with bold text
- Use color coding if needed: <span style="color: #dc2626;">AVOID</span> for contraindications

TONE:
- Professional but encouraging
- Empathetic to pain limitations
- Realistic about what's achievable
- Emphasize safety over speed of progress
- Motivational while being honest about challenges

Start directly with the content. Be specific, actionable, and tailored to this exact patient profile.
"""
        
        # Generate content using Gemini
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            return jsonify({
                'success': True,
                'plan': response.text
            })
        except Exception as gemini_error:
            error_message = str(gemini_error)
            if 'API_KEY_INVALID' in error_message:
                return jsonify({
                    'success': False,
                    'error': 'Invalid Gemini API key. Please check your API key and ensure it is activated for the Gemini API.'
                }), 400
            elif 'PERMISSION_DENIED' in error_message:
                return jsonify({
                    'success': False,
                    'error': 'API key does not have permission to use Gemini API. Please enable Generative Language API in Google Cloud Console.'
                }), 400
            elif 'RESOURCE_EXHAUSTED' in error_message:
                return jsonify({
                    'success': False,
                    'error': 'API quota exceeded. Please check your API usage limits or try again later.'
                }), 429
            else:
                return jsonify({
                    'success': False,
                    'error': f'Gemini API error: {error_message}'
                }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400



import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)



