from flask import Flask, render_template, request, jsonify
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
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCg_UZZmyvumvjDqbFfwP-U8aNJQgEeQG4')
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
- Pain Score: {pain_score}/10
- CRP Level: {crp} mg/L (inflammatory marker)
- Current Medication: {drug_type}

Instructions:
Generate a detailed, actionable plan with the following sections:

1. Diet Plan Overview - Brief summary of dietary goals (2-3 sentences)

2. Daily Meal Plan - Provide specific meals for:
   - Breakfast
   - Mid-morning Snack
   - Lunch
   - Evening Snack
   - Dinner
   
   For each meal, include:
   - Specific food items with approximate portions
   - Brief rationale (anti-inflammatory, joint health, weight management, etc.)

3. Key Nutritional Guidelines - List 5-6 important dietary principles:
   - Foods to emphasize (omega-3, antioxidants, etc.)
   - Foods to limit or avoid
   - Hydration goals
   - Supplements to consider (Vitamin D, Omega-3, etc.)

4. Exercise & Fitness Plan - Create a weekly exercise schedule:
   - Low-impact aerobic exercises (swimming, cycling, walking)
   - Strengthening exercises for knee support
   - Flexibility and range-of-motion exercises
   - Duration and frequency for each activity
   - Important precautions based on pain level and OA severity

5. Lifestyle Recommendations - Include:
   - Weight management strategies (if BMI indicates need)
   - Pain management techniques
   - Activity modifications
   - Sleep and stress management

6. Important Considerations - List:
   - Red flags or symptoms requiring immediate medical attention
   - How to progress exercises safely
   - When to adjust the plan

Critical Requirements:
- Tailor recommendations to the patient's OA severity and pain level
- Consider anti-inflammatory foods given CRP level of {crp} mg/L
- Account for {population} dietary preferences and patterns
- Provide gentle exercises if pain score is high
- Address smoking cessation if applicable
- Consider drug interactions with {drug_type}
- Be specific and actionable, not generic advice
- Keep the tone professional but encouraging

IMPORTANT FORMATTING RULES:
- Format your response in clean, professional HTML
- Use proper HTML tags: <h2> for main section headings, <h3> for subsections, <p> for paragraphs, <ul> and <li> for lists
- DO NOT use any asterisks (**), underscores (_), or markdown symbols
- DO NOT use markdown formatting at all
- Make the response easy to read for both doctors and patients
- Use clear headings and well-organized lists
- Start directly with the content, no preamble
"""
        
        # Generate content using Gemini
        try:
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
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


if __name__ == '__main__':
    app.run(debug=True)