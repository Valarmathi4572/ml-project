import joblib

class DummyScaler:
    def transform(self, X):
        return X

class DummyModel:
    def predict(self, X):
        # return 0 for each row
        return [0 for _ in X]

if __name__ == '__main__':
    joblib.dump(DummyModel(), 'diabetes_rf_smote_model.pkl')
    joblib.dump(DummyScaler(), 'scaler.pkl')
    print('Saved dummy model and scaler to current folder')
