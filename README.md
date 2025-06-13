# Neural Network from Scratch with NumPy

This repository contains a simple implementation of a feedforward neural network built from scratch using only **NumPy**. It supports training with mini-batch stochastic gradient descent and backpropagation.

The goal of this project is to understand the inner workings of neural networks without relying on high-level deep learning frameworks.

## References

This project was inspired by:

- [Michael Nielsen's free online book](http://neuralnetworksanddeeplearning.com/) – *Neural Networks and Deep Learning*
- DeepLearning.AI's Deep Learning Specialization on Coursera

Both resources were instrumental to build this implementation.

## Features

- Fully connected feedforward neural network
- Mini-batch stochastic gradient descent (SGD)
- Sigmoid activation
- Numpy-only (no TensorFlow, PyTorch, or Keras)

## 🚀 Getting Started

### Requirements

- Python 3.x
- NumPy

### Running the Code

Clone the repo:

```bash
git clone https://github.com/r8025n/neural-network.git
cd neural-network
```

Run the main script:

```bash
python3 main.py
```

You can modify the `main.py` file to adjust the network architecture, number of epochs, batch size, etc.

## 📁 Project Structure

```
.
├── main.py           # Entry point: trains the network
├── network.py        # Core NeuralNetwork class
├── utils.py          # Helper functions (e.g., sigmoid, sigmoid_prime)
├── data_loader.py    # Functions to load the MNIST dataset
├── data/             # Folder containing MNIST dataset in pickle format
│   └── mnist.pkl.gz  # MNIST dataset file (compressed)
└── README.md         
```
