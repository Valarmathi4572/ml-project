import joblib
import os

print("Checking model files...")
files = [f for f in os.listdir('.') if f.endswith('.pkl')]
print("PKL files found:", files)

print("\nTesting real model load...")
try:
    model = joblib.load('diabetes_rf_smote_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ SUCCESS: Real model loaded!")
    print("Model type:", type(model).__name__)
    print("Scaler type:", type(scaler).__name__)

    # Test with sample data
    sample_data = [[1, 85, 66, 29, 0, 26.6, 0.351, 31]]  # Sample diabetes data
    scaled_data = scaler.transform(sample_data)
    prediction = model.predict(scaled_data)
    print("Sample prediction result:", prediction[0])
    print("Prediction interpretation:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")

except Exception as e:
    print("❌ ERROR loading real model:", str(e))
    print("Using dummy model instead...")

    # Test dummy model
    try:
        from dummy_objects import DummyModel, DummyScaler
        dummy_model = DummyModel()
        dummy_scaler = DummyScaler()
        prediction = dummy_model.predict([[1,2,3,4,5,6,7,8]])
        print("Dummy model prediction:", prediction[0])
        print("Dummy prediction interpretation:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
    except Exception as e2:
        print("❌ ERROR with dummy model too:", str(e2))