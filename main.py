import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.load_data import load_fashion_mnist_data, load_california_housing_data
from training.train_classification import train_classification_model
from training.train_regression import train_regression_model
from training.train_wide_and_deep import train_wide_and_deep_model
from utils.plot import plot_samples, plot_history
from utils.tensorboard import setup_tensorboard_callbacks

def main():
    print("Welcome to the educational walkthrough of the machine learning project!\n")
    print("In this walkthrough, we will explore various models, visualize data, and utilize TensorBoard.\n")
    
    # Step 1: Load and Visualize Fashion MNIST Data
    print("Step 1: Loading and Visualizing Fashion MNIST Data...\n")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_fashion_mnist_data()
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    plot_samples(X_train, y_train, class_names)
    print("Visualized sample images from the Fashion MNIST dataset.\n")
    
    # Step 2: Train Classification Model
    print("Step 2: Training Classification Model on Fashion MNIST dataset...\n")
    print("The classification model is a neural network designed to recognize and categorize images of clothing items.\n")
    print("It consists of the following layers:\n")
    print("1. Input layer to specify the shape of the input images (28x28 pixels).\n")
    print("2. Flatten layer to convert each 28x28 image into a 1D array of 784 pixels.\n")
    print("3. Dense hidden layer with 300 neurons and ReLU activation function.\n")
    print("4. Dense hidden layer with 100 neurons and ReLU activation function.\n")
    print("5. Output layer with 10 neurons (one for each class) and softmax activation function.\n")
    model_cls, history_cls, X_test_cls, y_test_cls = train_classification_model()
    print("Classification Model Training Completed!\n")
    
    # Step 3: Load and Visualize California Housing Data
    print("Step 3: Loading and Visualizing California Housing Data...\n")
    X_train_h, y_train_h, X_valid_h, y_valid_h, X_test_h, y_test_h = load_california_housing_data()
    print("Loaded California Housing data. This dataset contains various features about houses in California,\n")
    print("such as median income, house age, average number of rooms, and geographical location.\n")
    print("Let's take a quick look at the first few records:\n")
    print(pd.DataFrame(X_train_h, columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']).head(), "\n")
    
    # Step 4: Train Regression Model
    print("Step 4: Training Regression Model on California Housing dataset...\n")
    print("The regression model is a neural network designed to predict the median house value based on input features.\n")
    print("It consists of the following layers:\n")
    print("1. Dense hidden layer with 30 neurons and ReLU activation function.\n")
    print("2. Output layer with 1 neuron to predict the house value.\n")
    model_reg, history_reg, X_test_reg, y_test_reg = train_regression_model()
    print("Regression Model Training Completed!\n")
    
    # Step 5: Train Wide and Deep Model
    print("Step 5: Training Wide and Deep Model on California Housing dataset...\n")
    print("The wide and deep model is a more complex neural network designed to improve prediction accuracy by combining a wide linear model and a deep neural network.\n")
    print("It consists of the following components:\n")
    print("1. Two Dense hidden layers with 30 neurons each and ReLU activation function for the deep part.\n")
    print("2. A concatenation layer to combine the deep part with the original input features (the wide part).\n")
    print("3. Two output layers: one for the main output (house value prediction) and one for auxiliary output (used to help the training process).\n")
    model_wd, history_wd, X_test_wd, y_test_wd = train_wide_and_deep_model()
    print("Wide and Deep Model Training Completed!\n")
    
    # Visualizations
    print("Let's visualize the training histories for the models:\n")
    plot_history(history_cls, title="Classification Model Training History")
    plot_history(history_reg, title="Regression Model Training History")
    plot_history(history_wd, title="Wide and Deep Model Training History")
    
    # TensorBoard
    print("\nYou can use TensorBoard to visualize the training logs for a more interactive experience.\n")
    print("To start TensorBoard, run the following command in your terminal:")
    print("tensorboard --logdir=./my_logs\n")
    print("Then open the provided URL in your web browser to explore the training metrics.\n")

if __name__ == "__main__":
    main()
