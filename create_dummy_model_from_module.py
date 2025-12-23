from dummy_objects import DummyModel, DummyScaler
import joblib

if __name__ == '__main__':
    joblib.dump(DummyModel(), 'diabetes_rf_smote_model.pkl')
    joblib.dump(DummyScaler(), 'scaler.pkl')
    print('Saved dummy model and scaler (from module) to current folder')
