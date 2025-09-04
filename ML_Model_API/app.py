from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print("Model load error:", str(e))
    model = None 
    scaler = None

full_features = ['Pregnancies', 'Glucose', 'BloodPressure','HighBP', 'HighChol', 'Smoker', 'Stroke', 'BMI', 'Age']
simple_features = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'BMI', 'Age']

@app.route('/api/full_details_prediction', methods=['POST'])
def full_details_prediction():
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        input_data = data['data']
        features = [input_data.get(feat, 0) for feat in full_features]
        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print("Error during full_details_prediction:", str(e))
        return jsonify({'error': 'ML API error (full_details_prediction)'}), 500

@app.route('/api/simple_details_prediction', methods=['POST'])
def simple_details_prediction():
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        input_data = data['data']
        extended_data = {
            'Pregnancies': 0,
            'Glucose': 0,
            'BloodPressure': 0,
            **input_data
        }
        features = [extended_data.get(feat, 0) for feat in full_features]
        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print("Error during simple_details_prediction:", str(e))
        return jsonify({'error': 'ML API error (simple_details_prediction)'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
