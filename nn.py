import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_architecture, hidden_activation=None, output_activation=None):
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.hidden_activation = hidden_activation or (lambda x: np.maximum(0, x)) 
        self.output_activation = output_activation or (lambda x: 1 if x > 0 else -1)

    def compute_num_weights(self):
        total_weights = 0
        input_size = self.input_size
        for n in self.hidden_architecture:
            total_weights += (input_size + 1) * n
            input_size = n
        total_weights += input_size + 1  
        return total_weights

    def load_weights(self, weights):
        w = np.array(weights)
        self.hidden_weights = []
        self.hidden_biases = []

        start = 0
        input_size = self.input_size

        if not self.hidden_architecture:
            total = len(w)
            for n in range(1, 100):  # tenta várias possibilidades
                needed = (input_size + 1) * n + n + input_size + 1
                if needed == total:
                    self.hidden_architecture = (n,)
                    break
                for m in range(1, 100):  # para 2 camadas ocultas
                    needed = (input_size + 1) * n + (n + 1) * m + m + input_size + 1
                    if needed == total:
                        self.hidden_architecture = (n, m)
                        break

        # reconstrói as camadas com a arquitetura especificada
        for n in self.hidden_architecture:
            end = start + (input_size + 1) * n
            layer = w[start:end]
            self.hidden_biases.append(layer[:n])
            self.hidden_weights.append(layer[n:].reshape(input_size, n))
            start = end
            input_size = n

        self.output_bias = w[start]
        self.output_weights = w[start + 1:]

    def forward(self, x):
        x = np.array(x)
        for w, b in zip(self.hidden_weights, self.hidden_biases):
            x = self.hidden_activation(np.dot(x, w) + b)
        return self.output_activation(np.dot(x, self.output_weights) + self.output_bias)


def create_network_architecture(input_size, hidden_layer=(16,)):
    return NeuralNetwork(input_size, hidden_layer)

