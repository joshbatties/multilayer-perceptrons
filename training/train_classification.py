import numpy as np
from data.load_data import load_fashion_mnist_data
from models.classification import create_classification_model
from utils.plot import plot_samples, plot_history

def train_classification_model():
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_fashion_mnist_data()
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    
    model = create_classification_model()
    
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid),)
    
    plot_history(history, title="Classification Model Training History")
    
    return model, history, X_test, y_test
