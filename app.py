from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from model import train_models, predict_disease

app = Flask(__name__)

# Train models on startup and store them
try:
    print("Training models...")
    models = train_models()
    print("Models trained successfully!")
except Exception as e:
    print(f"Error training models: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = pd.DataFrame([data])
        
        # Get predictions from all models
        results = predict_disease(features, models)
        
        return jsonify(results)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
