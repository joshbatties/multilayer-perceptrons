from data.load_data import load_california_housing_data
from models.wide_and_deep import create_wide_and_deep_model
from utils.plot import plot_history

def train_wide_and_deep_model():
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_california_housing_data()
    
    model = create_wide_and_deep_model()
    
    history = model.fit((X_train, X_train), (y_train, y_train), epochs=10,
                        validation_data=((X_valid, X_valid), (y_valid, y_valid)))
    
    plot_history(history, title="Wide and Deep Model Training History")
    
    return model, history, X_test, y_test
