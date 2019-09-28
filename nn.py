import numpy as np


class NeuralNetwork:
    def __init__(self, layers, inputs, outputs, epochs, eta=0.01, batch_size=64):
        """
        Initializes the neural network using random weights and biases.
        Weights are intialized using Xavier initialization.
        Biases are all intialized to zero.

        Parameters
        ----------
        layers: list, numpy array
            Number of neurons in each layer, including input and output layers
        inputs: list, numpy array
            Inputs to train the network on
        outputs: list, numpy array
            Outputs/labels to train the network on
        epochs: int
            Number of epochs to train the network for
        eta: float
            Learning rate of the neural network
        batch_size: int
            Number of training examples used to update weights
        """
        self.weights = None
        self.biases = None
        self.inputs = np.array(inputs, ndmin=2)
        self.outputs = np.array(outputs, ndmin=2)
        self.n_layers = len(layers)
        self.epochs = epochs
        self.eta = eta
        self.batch_size = batch_size

        # Check for valid input dimensions
        input_dimensions = self.inputs.shape
        if input_dimensions[1] != layers[0]:
            print("Incorrect input dimensions for the specified network architecture")
            self.inputs = np.array(inputs).reshape((-1, layers[0]))
            print(f"Input shape was {input_dimensions}, reshaped to {self.inputs.shape}")

        # Check for valid input dimensions
        output_dimensions = self.outputs.shape
        if output_dimensions[1] != layers[-1]:
            print("Incorrect output dimensions for the specified network architecture")
            self.outputs = np.array(outputs).reshape((-1, layers[-1]))
            print(f"Output shape was {output_dimensions}, reshaped to {self.outputs.shape}")

        if layers:
            self.weights = []
            for index, layer in enumerate(layers[:-1]):
                # Initialize weights using Xavier initialization
                self.weights.append(
                    np.random.normal(
                        loc=0,
                        scale=np.sqrt(2 / (layers[index] + layers[index + 1])),
                        size=(layers[index], layers[index + 1]),
                    )
                )
            # Initialize biases to zero
            self.biases = np.zeros((len(layers) - 1, 1))

    def feed_forward(self):
        """
        Feed forward the inputs through the different layers
        of the neural network.

        Returns
        -------
            Activations for each layer, including the input values
        """
        activations = [self.inputs]
        # Store all zs and activations
        for W, b in zip(self.weights, self.biases):
            activations.append(self.sigmoid(np.dot(activations[-1], W) + b))
        return activations

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters
        ----------
        x: int, list, numpy array
            Input value(s) for the sigmoid function

        Returns
        -------
            Sigmoid of the input value(s)
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Returns the derivative of the sigmoid function (da/dz).

        Parameters
        ----------
        x: int, list, numpy array
            Input value(s) for the sigmoid derivative

        Returns
        -------
            Sigmoid derivative of the input value(s)
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def cost_function(self, y_hat, y):
        """
        Returns the cost using the squared error as the cost function.

        Parameters
        ----------
        y_hat: int, list, numpy array
            Output value(s) as predicted by the neural network
        y: int, list, numpy array
            Actual output value(s)

        Returns
        -------
            Squared error of the prediction
        -------
        """
        return (y_hat - y) ** 2

    def cost_derivative(self, y_hat, y):
        """
        Returns the derivative of the squared error cost function.

        Parameters
        ----------
        y_hat: int, list, numpy array
            Output value(s) as predicted by the neural network
        y: int, list, numpy array
            Actual output value(s)

        Returns
        -------
            Derivative of the cost function
        """
        return 2 * (y_hat - y)

    def back_propagation(self, inputs, outputs):
        """
        Returns the weights and biases gradients with respect
        to the cost function (dC/dW).

        Returns
        -------
        dCdW: Weight gradients
        dCdb: Bias gradients
        """
        zs = []
        activations = [inputs]
        # Store all zs and activations
        for W, b in zip(self.weights, self.biases):
            zs.append(np.dot(activations[-1], W) + b)
            activations.append(self.sigmoid(zs[-1]))
        prediction = activations[-1]
        dCdW = [np.zeros(w.shape) for w in self.weights]
        dCdb = [np.zeros(b.shape) for b in self.biases]
        dCdW[-1] = self.sigmoid_derivative(prediction) * self.cost_derivative(prediction, outputs)

        # Back propagate the hidden layers
        for i in range(1, self.n_layers):
            delta = self.sigmoid_derivative(zs[-i]) * self.cost_derivative(prediction, outputs)
            dCdW[-i] = np.dot(activations[-i - 1].transpose(), delta)
            dCdb[-i] = np.sum(1 * delta)
        return dCdW, dCdb

    def gradient_descent(self, inputs, outputs, eta):
        """
        Updates weights and biases using gradient descent.
        """
        gradient_weights, gradient_biases = self.back_propagation(inputs, outputs)
        for W, b, dW, db in zip(self.weights, self.biases, gradient_weights, gradient_biases):
            W += -eta * dW
            b += -eta * db

    def train(self):
        for epoch in range(1, self.epochs + 1):
            # Combine input and output data
            data = np.concatenate((self.inputs, self.outputs), axis=1)
            # Randomly shuffle data
            np.random.shuffle(data)
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i : i + self.batch_size]
                inputs, outputs = batch[:, : -layers[-1]], batch[:, -layers[-1] :]
                self.gradient_descent(inputs, outputs, self.eta)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, loss: {self.cost_function(self.feed_forward()[-1], self.outputs).mean()}"
                )

    def predict(self, inputs):
        """
        Predict label(s) for given inputs.

        Parameters
        ----------
        inputs: int, list, numpy array
            Inputs to use for the prediction
        """
        activations = [inputs]
        # Store all zs and activations
        for W, b in zip(self.weights, self.biases):
            activations.append(self.sigmoid(np.dot(activations[-1], W) + b))
        return activations[-1]


if __name__ == "__main__":

    # Create data using the sigmoid function
    n_observations = 100
    n_features = 2
    np.random.seed(seed=1)
    x = np.random.random((n_observations, n_features))
    y = 1 / (
        1
        + np.exp(
            # Multiply inputs with specified weights
            -(
                np.dot(x, np.array([[-1, 2]]).transpose())
                # Add noise
                + np.random.normal(size=(n_observations, 1)) / 100
                # Add bias
                - 0.8
            )
        )
    )

    # Train network
    layers = [2, 1]
    nn = NeuralNetwork(layers=layers, inputs=x, outputs=y, epochs=1000, eta=0.1)
    nn.train()
    # See trained weights and biases
    print(nn.weights)
    print(nn.biases)
