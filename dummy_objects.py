class DummyScaler:
    def transform(self, X):
        return X

class DummyModel:
    def predict(self, X):
        return [0 for _ in X]
