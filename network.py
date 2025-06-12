import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

class Network:
    def __init__(self, sizes):
        num_layers: len(sizes)
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
    
