import numpy as np
from utils import sigmoid, sigmoid_prime 
import random

class Network:
    def __init__(self, sizes):
        self.num_layers: len(sizes)
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

    def feed_forward(self, input):
        input = np.array(input)

        for w, b in zip(self.weights, self.biases):
            print("w shape:", w.shape)
            print("input shape:", input.shape)
            input = sigmoid(np.dot(w, input))
            print("w.b dot product shape:", input.shape)
            print("----------------------")
        
        return input

    def learn(training_data, epochs, learning_rate, batch_size, test_data=None):
        # will learn via gradient descent
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k : k + batch_size] for k in range(0, len(training_data), batch_size)]
            
            for mini_batch in mini_batches:
                update_weights(mini_batch)

        return
    
    # def update_weights(mini_batch):
    #     delta_w = [np.zeros(w.shape) for w in self.weights]
    #     delta_b = [np.zeros(b.shape) for b in self.weights]

    #     for x, y in mini_batch:
    #         gradient_w, gradient_b = backpropagate(x, y)


    #     return

    def backpropagate(x, y):
        """ lets assume 3 nodes. 1 input, 1 output,
        1 hidden in the middle.
        So the computation graph for the forward pass flow should be
        x -> z1 -> a1 -> z2 -> a2 -> C.
        z1 = w1x + b1, a1 = sigma(z1), z2 = w2a1 + b2, a2 = sigma(z2),
        C = 1/2(a2 - y)^2 
        Backward pass flow will be just reverse
        In the following code dx means dC/dx""" 

        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]

        # ---------------forward pass-------------------
        activation = x
        activations = [] # List to store activations layer by layer
        zs = []           # List to store z = w·a + b for each layer
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # --------------backward pass--------------------
        dz = self.cost_derivative(activations[-1], y) * utils.sigmoid_prime(zs[-1])
        gradient_b[-1] = dz
        gradient_w[-1] = np.dot(dz, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            dz = np.dot(self.weights[-layer + 1].transpose(), dz) * sigmoid_prime(zs[-layer])
            gradient_b[-layer] = dz
            gradient_w[-layer] = np.dot(dz, activations[-layer-1].transpose())


        return (gradient_w, gradient_b)

    def cost_derivative(output_activations, y):
        return (output_activations - y) # this is the derivative of qudratic mse cost function

    
    
    
