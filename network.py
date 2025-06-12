import numpy as np

class Network:
    def __init__(self, sizes):
        num_layers: len(sizes)
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        self.bias = [np.random.randn(x, 1) for x in sizes[1:]]

    
