import numpy as np
import random

from utils import sigmoid, sigmoid_prime 

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

    def feed_forward(self, input):
        input = np.array(input)

        for w, b in zip(self.weights, self.biases):
            input = sigmoid(np.dot(w, input))

        return input

    def learn(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        # learn via stochastic gradient descent
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k : k + batch_size] for k in range(0, len(training_data), batch_size)]
            
            for mini_batch in mini_batches:
                self.update_weights(mini_batch, learning_rate)
            
            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(j))

        return
    
    def update_weights(self, mini_batch, learning_rate):
        delta_w_total = [np.zeros(w.shape) for w in self.weights]
        delta_b_total = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_w, delta_b = self.backpropagate(x, y)
            delta_w_total = [dwt + dw for dwt, dw in zip(delta_w_total, delta_w)]
            delta_b_total = [dbt + db for dbt, db in zip(delta_b_total, delta_b)]

        delta_w_avg = [x/len(mini_batch) for x in delta_w_total] 
        delta_b_avg = [x/len(mini_batch) for x in delta_b_total] 

        self.weights = [w - (learning_rate * dw) for w, dw in zip(self.weights, delta_w_avg)]
        self.biases = [b - (learning_rate * db) for b, db in zip(self.biases, delta_b_avg)]

        return

    def backpropagate(self, x, y):
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
        activations = [x] # List to store activations layer by layer
        zs = []           # List to store z = w·a + b for each layer
        
        for w, b in zip(self.weights, self.biases):
            # print("activation shape", activation.shape)
            z = np.dot(w, activation) + b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # --------------backward pass--------------------
        dz = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        gradient_b[-1] = dz
        gradient_w[-1] = np.dot(dz, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            # print("layer", layer)
            dz = np.dot(self.weights[-layer + 1].transpose(), dz) * sigmoid_prime(zs[-layer])
            gradient_b[-layer] = dz
            gradient_w[-layer] = np.dot(dz, activations[-layer-1].transpose())

        return (gradient_w, gradient_b)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y) # this is the derivative of qudratic mse cost function

    def evaluate(self, test_data):
        match_count = 0

        for x, y in test_data:
            output = self.feed_forward(x)
            y_prime = np.argmax(output)
            match_count = match_count + int(y_prime == y)

        return match_count
    
    
