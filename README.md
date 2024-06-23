# multi-layer perceptrons

This project provides a comprehensive demonstration of training various neural network models using TensorFlow and Keras. The project includes three models: an image classification model, a regression model, and a wide-and-deep model. Through this project, we will be loading and preprocessing data, designing and implementing neural network models, training these models, and evaluating their performance. Additionally, the project includes visualisation techniques to better understand the training processes and the results. The overall goal is to provide a comprehensive learning experience in neural network training and application using TensorFlow and Keras.

## Installation
To set up the project, clone the repository and install the required dependencies using pip.

```bash
git clone https://github.com/yourusername/neural-network-training.git
cd neural-network-training
pip install -r requirements.txt
```

## Usage
Run the main script to train the models and visualize the results.

```bash
python main.py
```
## Project Structure
```css
neural-network-training/
├── data/
│   └── load_data.py
├── models/
│   ├── classification.py
│   ├── regression.py
│   └── wide_and_deep.py
├── training/
│   ├── train_classification.py
│   ├── train_regression.py
│   └── train_wide_and_deep.py
├── utils/
│   └── plot.py
├── main.py
├── requirements.txt
└── README.md
```
## Models
### Classification Model
The classification model is designed to recognize and categorize images of clothing items from the Fashion MNIST dataset. This model employs a simple feedforward neural network architecture, making it suitable for beginner-level tasks in computer vision.

Input Layer: Accepts grayscale images of size 28x28 pixels.
Flatten Layer: Converts the 2D image into a 1D array of 784 pixels.
Dense Layer 1: Contains 300 neurons with ReLU activation, capable of capturing complex patterns in the data.
Dense Layer 2: Contains 100 neurons with ReLU activation, adding another layer of abstraction.
Output Layer: Comprises 10 neurons with softmax activation, each representing a class of clothing items (e.g., T-shirt, trousers).
This model is trained using the sparse_categorical_crossentropy loss function and the Stochastic Gradient Descent (SGD) optimizer.

### Regression Model
The regression model is designed to predict the median house value from various features in the California Housing dataset. It utilizes a straightforward architecture suitable for regression tasks, capturing linear and non-linear relationships in the data.

Input Layer: Handles eight numerical features, such as median income, house age, and average number of rooms.
Dense Layer: Consists of 30 neurons with ReLU activation, providing the capacity to model complex relationships.
Output Layer: Contains a single neuron without activation, representing the predicted house value.
The model is trained using the mean_squared_error loss function and the SGD optimizer with a learning rate of 0.001.

### Wide and Deep Model
The wide and deep model combines the strengths of linear models and deep neural networks to enhance prediction accuracy for complex datasets. This hybrid approach captures both low-level interactions and high-level patterns.

Input Layers: Accepts features for both wide and deep components. The wide component captures the raw input features, while the deep component models complex interactions.
Deep Part: Includes two dense layers, each with 30 neurons and ReLU activation, providing a deep representation of the input features.
Concatenation Layer: Merges the outputs from the deep part and the original input features.
Output Layers: Includes a main output layer for the primary prediction task (house value) and an auxiliary output layer to assist in training by providing an additional signal.
This model is trained using the mean_squared_error loss function with loss weights for the main and auxiliary outputs and the SGD optimizer with a learning rate of 0.001.

## Data
Fashion MNIST
This dataset contains 70,000 grayscale images divided into 60,000 training images and 10,000 test images. Each image is 28x28 pixels, representing 10 different categories of clothing items such as T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.

California Housing
This dataset is derived from the 1990 California census and contains various features about houses, including median income, house age, average number of rooms, average number of bedrooms, population, average occupancy, latitude, and longitude. The target variable is the median house value, making this dataset suitable for regression analysis.

## Training Scripts
train_classification.py
Trains the classification model on the Fashion MNIST dataset and plots the training history.

train_regression.py
Trains the regression model on the California Housing dataset and plots the training history.

train_wide_and_deep.py
Trains the wide and deep model on the California Housing dataset and plots the training history.

## Visualization
The utils/plot.py module provides functions for plotting training histories and visualizing sample images from the Fashion MNIST dataset.

Plot Training History
Plots the training and validation accuracy/loss over epochs for each model.

Plot Sample Images
Displays a grid of sample images from the Fashion MNIST dataset with their corresponding class names.
