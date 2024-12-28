from flask import Flask, render_template, request, jsonify
import numpy as np
from main import MedicalPredictionSystem
import json

app = Flask(__name__)
predictor = MedicalPredictionSystem()

# Load the model on startup
try:
    predictor.load_model()
except FileNotFoundError:
    print("No pre-trained model found. Training new model...")
    predictor.train_new_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert data to numpy array
        patient_data = np.array([list(data.values())]).reshape(1, -1)
        
        # Make prediction
        prediction, probability = predictor.predict(patient_data)
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'risk_level': 'High' if probability[0][1] > 0.7 else 'Medium' if probability[0][1] > 0.3 else 'Low'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        predictor.train_new_model()
        return jsonify({'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info')
def model_info():
    return jsonify({
        'features': predictor.get_feature_importance(),
        'performance_metrics': predictor.get_performance_metrics(),
        'last_trained': predictor.get_last_trained_date()
    })

if __name__ == '__main__':
    app.run(debug=True)
