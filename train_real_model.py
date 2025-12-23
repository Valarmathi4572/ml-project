import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

print("🔄 Training real diabetes prediction model...")

# Load the diabetes dataset
df = pd.read_csv('diabetes.csv')
print(f"Dataset shape: {df.shape}")
print(f"Outcome distribution:\n{df['Outcome'].value_counts()}")

# Handle zero values (same as in your notebook)
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    df[col] = df[col].replace(0, df[col].mean())

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Features: {list(X.columns)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"Original training set: {X_train.shape[0]} samples")
print(f"After SMOTE: {X_train_smote.shape[0]} samples")
print(f"SMOTE class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

print("Training Random Forest model...")
rf_model.fit(X_train_smote, y_train_smote)

# Evaluate on test set
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and scaler
joblib.dump(rf_model, 'diabetes_rf_smote_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n✅ Model and scaler saved successfully!")
print("📁 Files saved:")
print("   - diabetes_rf_smote_model.pkl (Random Forest with SMOTE)")
print("   - scaler.pkl (StandardScaler)")

# Test the saved model
print("\n🧪 Testing saved model...")
loaded_model = joblib.load('diabetes_rf_smote_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Test with sample data
sample_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  # High risk sample
scaled_sample = loaded_scaler.transform(sample_data)
prediction = loaded_model.predict(scaled_sample)
probability = loaded_model.predict_proba(scaled_sample)

print(f"Sample input: {sample_data[0]}")
print(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
print(f"Confidence - Non-Diabetic: {probability[0][0]:.3f}, Diabetic: {probability[0][1]:.3f}")

# Test with low risk sample
low_risk_sample = [[1, 85, 66, 29, 0, 26.6, 0.351, 31]]
scaled_low = loaded_scaler.transform(low_risk_sample)
pred_low = loaded_model.predict(scaled_low)
prob_low = loaded_model.predict_proba(scaled_low)

print(f"\nLow risk sample: {low_risk_sample[0]}")
print(f"Prediction: {'Diabetic' if pred_low[0] == 1 else 'Non-Diabetic'}")
print(f"Confidence - Non-Diabetic: {prob_low[0][0]:.3f}, Diabetic: {prob_low[0][1]:.3f}")

print("\n🎉 Real prediction model is ready! Restart your Flask app to use it.")