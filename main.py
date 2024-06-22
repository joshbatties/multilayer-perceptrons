import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.load_data import load_fashion_mnist_data, load_california_housing_data
from training.train_classification import train_classification_model
from training.train_regression import train_regression_model
from training.train_wide_and_deep import train_wide_and_deep_model
from utils.plot import plot_samples, plot_history

def main():
    print("We will explore 3 different types of neural networks and train them to classify images and predict values.\n")
    
    # Step 1: Load and Visualize Fashion MNIST Data
    print("Step 1: Loading and Visualizing Fashion MNIST Data...\n")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_fashion_mnist_data()
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    plot_samples(X_train, y_train, class_names)
    print("Visualized sample images from the Fashion MNIST dataset.\n")
    
    # Step 2: Train Classification Model
    print("Step 2: Training Classification Model on Fashion MNIST dataset...\n")
    print("The classification model is a neural network designed to recognize and categorize images of clothing items.")
    print("Model Details:")
    print("1. Input: Images of size 28x28 pixels.")
    print("2. Flatten Layer: Converts each 28x28 image into a 1D array of 784 pixels.")
    print("3. Dense Layer 1: 300 neurons with ReLU activation function.")
    print("4. Dense Layer 2: 100 neurons with ReLU activation function.")
    print("5. Output Layer: 10 neurons with softmax activation function, one for each class.\n")
    model_cls, history_cls, X_test_cls, y_test_cls = train_classification_model()
    print("Classification Model Training Completed!\n")

    # Evaluate the classification model
    loss_cls, accuracy_cls = model_cls.evaluate(X_test_cls, y_test_cls, verbose=0)
    print(f"Classification Model Accuracy: {accuracy_cls * 100:.2f}%")
    print(f"The classification model achieves an accuracy of {accuracy_cls * 100:.2f}%. This means that when the model is given images it hasn't seen before, it correctly identifies the type of clothing item {accuracy_cls * 100:.2f}% of the time.")
    print("For example, if the model is shown an image of a sneaker, it will correctly classify it as a sneaker about 90% of the time, assuming an accuracy of 90%. This level of accuracy is considered quite good for image classification tasks.\n")
    
    # Step 3: Load and Visualize California Housing Data
    print("Step 3: Loading and Visualizing California Housing Data...\n")
    X_train_h, y_train_h, X_valid_h, y_valid_h, X_test_h, y_test_h = load_california_housing_data()
    print("Loaded California Housing data. This dataset contains various features about houses in California,")
    print("such as median income, house age, average number of rooms, and geographical location.")
    print("Let's take a quick look at the first few records:\n")
    print(pd.DataFrame(X_train_h, columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']).head(), "\n")
    
    # Step 4: Train Regression Model
    print("Step 4: Training Regression Model on California Housing dataset...\n")
    print("The regression model is a neural network designed to predict the median house value based on input features.")
    print("Model Details:")
    print("1. Input: Features such as median income, house age, etc.")
    print("2. Dense Layer: 30 neurons with ReLU activation function.")
    print("3. Output Layer: 1 neuron to predict the house value.\n")
    model_reg, history_reg, X_test_reg, y_test_reg = train_regression_model()
    print("Regression Model Training Completed!\n")
    
    # Evaluate the regression model
    mse_reg = model_reg.evaluate(X_test_reg, y_test_reg, verbose=0)
    print(f"Regression Model Mean Squared Error: {mse_reg:.2f}")
    print(f"The regression model predicts the median house value with a mean squared error (MSE) of {mse_reg:.2f}.")
    print("Mean Squared Error is a measure of how close the model's predictions are to the actual house values. A lower MSE indicates better performance. For example, if the MSE is 0.25, it means that, on average, the square of the difference between the predicted and actual house values is 0.25. This helps us understand the accuracy of the model's predictions.\n")
    
    # Step 5: Train Wide and Deep Model
    print("Step 5: Training Wide and Deep Model on California Housing dataset...\n")
    print("The wide and deep model is a more complex neural network designed to improve prediction accuracy by combining a wide linear model and a deep neural network.")
    print("Model Details:")
    print("1. Input: Features such as median income, house age, etc.")
    print("2. Deep Part: Two Dense layers with 30 neurons each and ReLU activation function.")
    print("3. Concatenation Layer: Combines the deep part with the original input features (wide part).")
    print("4. Output Layers: One for the main output (house value prediction) and one for auxiliary output to help the training process.\n")
    model_wd, history_wd, X_test_wd, y_test_wd = train_wide_and_deep_model()
    print("Wide and Deep Model Training Completed!\n")
    
    # Evaluate the wide and deep model
    mse_wd = model_wd.evaluate(X_test_wd, y_test_wd, verbose=0)
    print(f"Wide and Deep Model Mean Squared Error: {mse_wd:.2f}")
    print(f"The wide and deep model predicts the median house value with a mean squared error (MSE) of {mse_wd:.2f}.")
    print("Like the regression model, the MSE here indicates how close the predictions are to the actual values. The wide and deep model is designed to capture more complex relationships in the data, which can lead to more accurate predictions. A lower MSE compared to the standard regression model suggests that this approach is better at predicting house values in this dataset.\n")
    
    # Visualizations
    print("Let's visualize the training histories for the models:\n")
    plot_history(history_cls, title="Classification Model Training History")
    plot_history(history_reg, title="Regression Model Training History")
    plot_history(history_wd, title="Wide and Deep Model Training History")

if __name__ == "__main__":
    main()
