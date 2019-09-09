import numpy as np


class NeuralNetwork:
    def __init__(self, layers=None):
        """
        Initializes the neural network using random weights and biases.
        Weights are intialized using Xavier initialization.
        Biases are all intialized to zero.


        Parameters
        ----------
        layers: list
            Number of neurons in each layer, including input and output layers
        """
        self.weights = None
        self.biases = None
        if layers:
            self.weights = []
            for index, layer in enumerate(layers[1:]):
                # Initialize weights using Xavier initialization
                self.weights.append(
                    np.random.normal(
                        loc=0,
                        scale=np.sqrt(2 / (layers[index] + layers[index + 1])),
                        size=(layers[index], layers[index + 1]),
                    )
                )
            # Initialize biases to zero
            self.biases = np.zeros(len(layers) - 1)


if __name__ == "__main__":
    layers = [2, 4, 2]
    nn = NeuralNetwork(layers=layers)
