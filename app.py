from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

print("=" * 60)
print("CRIME PREDICTION API - RENDER DEPLOYMENT")
print("=" * 60)

# Load your trained model components
try:
    pipeline = joblib.load('crime_xgb_onehot_weighted_v1.pkl')
    target_encoder = joblib.load('target_label_encoder_v1.pkl')
    metadata = joblib.load('crime_onehot_weighted_metadata_v1.pkl')

    categorical_cols = metadata['categorical_cols']
    numeric_cols = metadata['numeric_cols']
    feature_cols = metadata['feature_cols']
    class_names = metadata['class_names']
    custom_weights = metadata['custom_weights']

    print("✓ Model loaded successfully!")
    print(f"✓ Classes: {class_names}")
    print(f"✓ Weights: {custom_weights}")

except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)


class FinalCrimePredictor:
    def __init__(self, pipeline, target_encoder, categorical_cols, numeric_cols, feature_cols, class_names):
        self.pipeline = pipeline
        self.target_encoder = target_encoder
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.feature_cols = feature_cols
        self.class_names = class_names

    def preprocess(self, input_data):
        df = pd.DataFrame([input_data])
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().fillna("unknown")
            else:
                df[col] = "unknown"
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            else:
                df[col] = 0.0
        return df[self.feature_cols]

    def predict(self, input_data):
        X = self.preprocess(input_data)
        pred_encoded = self.pipeline.predict(X)[0]
        pred_label = self.target_encoder.inverse_transform([pred_encoded])[0]
        probabilities = self.pipeline.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3 = []
        for idx in top3_idx:
            top3.append({
                'crime_type': self.class_names[idx],
                'probability': float(probabilities[idx])
            })
        return {
            'prediction': {
                'crime_type': pred_label,
                'confidence': confidence,
                'class_index': int(pred_encoded)
            },
            'all_probabilities': {
                name: float(prob) for name, prob in zip(self.class_names, probabilities)
            },
            'top3_predictions': top3,
            'input_summary': {
                'location': f"{input_data.get('incident_place', '')}, {input_data.get('incident_district', '')}",
                'time': f"Week {input_data.get('incident_week', '')}, {input_data.get('incident_weekday', '')} {input_data.get('part_of_the_day', '')}"
            }
        }


predictor = FinalCrimePredictor(
    pipeline, target_encoder,
    categorical_cols, numeric_cols, feature_cols,
    class_names
)


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Crime Prediction API',
        'status': 'active',
        'classes': class_names,
        'weights': custom_weights
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'classes': class_names,
        'weights': custom_weights
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        result = predictor.predict(data)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)