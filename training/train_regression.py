from data.load_data import load_california_housing_data
from models.regression import create_regression_model
from utils.plot import plot_history
from utils.tensorboard import setup_tensorboard_callbacks

def train_regression_model():
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_california_housing_data()
    
    model = create_regression_model(X_train.shape[1:])
    tensorboard_cb, run_logdir = setup_tensorboard_callbacks()
    
    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_valid, y_valid),
                        callbacks=[tensorboard_cb])
    
    plot_history(history, title="Regression Model Training History")
    
    return model, history, X_test, y_test
