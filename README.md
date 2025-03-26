# 🧠 ANNMNIST - Artificial Neural Network for MNIST Digit Recognition

A simple implementation of a neural network for recognizing handwritten digits from the MNIST dataset using Keras.

## 📝 Description

This repository contains a Python implementation of a neural network designed to classify handwritten digits from the MNIST dataset. It demonstrates fundamental concepts in deep learning including data preprocessing, network architecture design, training, evaluation, and visualization techniques.

## ✨ Features

- 📊 Loads and preprocesses the MNIST dataset
- 🏗️ Builds a sequential neural network with dense layers
- 🔄 Trains the network using backpropagation
- 📈 Evaluates model performance on test data
- 👁️ Visualizes digit samples and demonstrates tensor slicing techniques
- ⏱️ Includes timing functionality to measure performance

## 🔧 Prerequisites

- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy

## 🚀 Setup Guide

1. Clone this repository:
   ```bash
   git clone https://github.com/corticalstack/ANNMNIST.git
   cd ANNMNIST
   ```

2. Install the required dependencies:
   ```bash
   pip install tensorflow keras matplotlib numpy
   ```

## 💻 Usage

Run the main script to load the MNIST dataset, build and train the neural network, and evaluate its performance:

```bash
python main.py
```

The script will:
1. Load the MNIST dataset
2. Display sample digits from the dataset
3. Demonstrate tensor slicing techniques on sample digits
4. Build and train a neural network
5. Evaluate the network's performance on the test dataset

## 🧩 Architecture

The neural network architecture consists of:
- Input layer: Flattened 28x28 pixel images (784 input neurons)
- Hidden layer: 512 neurons with ReLU activation
- Output layer: 10 neurons (one for each digit 0-9) with softmax activation

The model is compiled with:
- Optimizer: RMSprop
- Loss function: Categorical cross-entropy
- Metric: Accuracy

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
