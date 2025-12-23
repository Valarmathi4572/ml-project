import pickle

def load_model():
    """Load and return the trained ML model"""
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
