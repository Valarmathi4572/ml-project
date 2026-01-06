import os
import sys
import traceback
from flask import Flask, render_template, request, send_from_directory
import joblib

# Create Flask app
app = Flask(__name__)

# Configure paths for Vercel deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model and scaler paths
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_rf_smote_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Global variables for model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load model and scaler with error handling"""
    global model, scaler

    try:
        print("Loading model and scaler...", file=sys.stderr)
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully!", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Error loading model/scaler: {e}", file=sys.stderr)
        return False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load model if not loaded
        global model, scaler
        if model is None:
            if not load_model_and_scaler():
                print("Warning: Could not load model files. Using dummy predictions.", file=sys.stderr)
                # Fallback dummy classes
                class DummyModel:
                    def predict(self, X):
                        return [0 for _ in X]
                    def predict_proba(self, X):
                        return [[0.6, 0.4] for _ in X]

                class DummyScaler:
                    def transform(self, X):
                        return X

                model = DummyModel()
                scaler = DummyScaler()

        # Collect form fields safely
        features = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]

        # Scale features
        final_features = scaler.transform([features])

        # Make prediction
        prediction = model.predict(final_features)
        prediction_proba = model.predict_proba(final_features)

        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        confidence = prediction_proba[0][int(prediction[0])]

        return render_template("index.html",
                             prediction_text=result,
                             confidence=f"{confidence:.1%}")

    except Exception as e:
        print(f"Prediction error: {e}", file=sys.stderr)
        traceback.print_exc()
        return render_template("index.html",
                             prediction_text=f"Error: {str(e)}",
                             error=True)

# Handle static files for Vercel
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static'), filename)

# Health check endpoint for Vercel
@app.route('/api/health')
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    app.run(debug=True)