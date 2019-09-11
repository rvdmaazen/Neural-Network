import numpy as np


class NeuralNetwork:
    def __init__(self, layers=None, inputs=None):
        """
        Initializes the neural network using random weights and biases.
        Weights are intialized using Xavier initialization.
        Biases are all intialized to zero.

        Parameters
        ----------
        layers: list
            Number of neurons in each layer, including input and output layers
        inputs: list
            Inputs to train the network on
        """
        self.weights = None
        self.biases = None
        self.inputs = np.array(inputs, ndmin=2)

        # Check for valid input dimensions
        input_dimensions = self.inputs.shape
        if input_dimensions[1] != layers[0]:
            print("Incorrect input dimensions for the specified network architecture")
            self.inputs = np.array(inputs).reshape((-1, layers[0]))
            print(f"Input shape was {input_dimensions}, reshaped to {self.inputs.shape}")

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

    def feed_forward(self):
        output = self.inputs
        for i, W in enumerate(self.weights):
            output = self.sigmoid(np.dot(output, W) + self.biases[i])
        self.output = output

    def sigmoid(self, x):
        """
        Sigmoid activation function

        Parameters
        ----------
        x: int, list, numpy array
            Input value(s) for the sigmoid function

        Returns
        -------
        y: int, list, numpy array
        """
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    layers = [2, 4, 2]
    inputs = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    nn = NeuralNetwork(layers=layers, inputs=inputs)
    nn.feed_forward()
