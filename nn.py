import numpy as np
from activations_functions import ACTIVATION_FUNCTIONS
from loss_functions import LOSS_FUNCTIONS


class NeuralNetwork:
    def __init__(self, layers, inputs, outputs, epochs, activations, loss, eta=0.01, batch_size=64):
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
        activations: list
            Activation functions to use
        eta: float
            Learning rate of the neural network
        batch_size: int
            Number of training examples used to update weights
        """
        self.weights = None
        self.biases = None
        self.inputs = np.array(inputs, ndmin=2)
        self.outputs = np.array(outputs, ndmin=2)
        self.layers = layers
        self.n_layers = len(self.layers)
        self.epochs = epochs
        self.eta = eta
        self.batch_size = batch_size
        self.activation_types = activations
        self.loss = loss

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
            self.biases = []
            self.weights = []
            for index, layer in enumerate(layers[:-1]):
                # Initialize weights using Xavier initialization
                self.weights.append(
                    np.random.normal(
                        loc=0,
                        scale=np.sqrt(2 / (layers[index] + layers[index + 1])),
                        size=(layers[index + 1], layers[index])
                    ).T
                )
                # Initialize biases to zero
                self.biases.append(np.zeros((layers[index + 1], 1)).T) 

        # Check for correct number of activation functions
        if len(self.activation_types) != self.n_layers - 1:
            raise ValueError(
                f"Incorrect number of activation functions specficied: expected {self.n_layers - 1}, got {len(self.activation_types)}"
            )

        # Check for valid activation functions
        for activation in self.activation_types:
            if activation not in ACTIVATION_FUNCTIONS:
                raise Exception(f"Unknown activation function {activation}")
        self.activations = [ACTIVATION_FUNCTIONS[activation]() for activation in self.activation_types]

        # Check for valid loss function
        if self.loss not in LOSS_FUNCTIONS:
            raise(Exception(f"Unknow loss function {self.loss}"))
        self.loss_function = LOSS_FUNCTIONS[self.loss]()

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
        for W, b, activation_type in zip(self.weights, self.biases, self.activations):
            activations.append(self.activation_function(np.dot(activations[-1], W) + b, activation_type))
        return activations

    def activation_function(self, x, activation):
        """
        Returns the activations calculated using the specified activation function.

        Parameters
        ----------
        x: int, list, numpy array
            Input value(s) for the activation function
        activation: class
            Activation class to use

        Returns
        -------
            Activation of the input value(s)
        """
        return activation.function(x)

    def activation_derivative(self, x, activation):
        """
        Returns the derivative of the specified activation function (da/dz).

        Parameters
        ----------
        x: int, list, numpy array
            Input value(s) for the activation derivative
        activation: class
            Activation class to use

        Returns
        -------
            Activation derivative of the input value(s)
        """
        return activation.derivative(x)

    def cost_function(self, y_hat, y):
        """
        Returns the cost using the squared error as the cost function (dC/da).

        Parameters
        ----------
        y_hat: int, list, numpy array
            Output value(s) as predicted by the neural network
        y: int, list, numpy array
            Actual output value(s)

        Returns
        -------
            Squared error of the prediction
        """
        return self.loss_function.loss(y_hat, y)

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
        return self.loss_function.derivative(y_hat, y)

    def back_propagation(self, inputs, outputs):
        """
        Returns the weights and biases gradients with respect
        to the cost function (dC/dW).

        Returns
        -------
        dCdW: numpy array
            Weight gradients
        dCdb: numpy array
            Bias gradients
        """
        m = self.inputs.shape[0]
        zs = []
        activations = [inputs]
        # Store all zs and activations based on old weights and biases
        for W, b, activation_type in zip(self.weights, self.biases, self.activations):
            zs.append(np.dot(activations[-1], W) + b)
            activations.append(self.activation_function(zs[-1], activation_type))

        # Create empty arrays for weight and bias gradients
        dCdW = [np.zeros(w.shape) for w in self.weights]
        dCdb = [np.zeros(b.shape) for b in self.biases]

        # Back propagate the output layer
        delta = self.activation_derivative(zs[-1], self.activations[-1]) * self.cost_derivative(activations[-1], outputs)
        dCdW[-1] = np.dot(activations[-2].T, delta) / m
        dCdb[-1] = np.sum(1 * delta, axis=0) / m

        # Back propagate the hidden layers
        for i in range(2, self.n_layers):
            delta = np.dot(delta, self.weights[-i + 1].T) * self.activation_derivative(zs[-i], self.activations[-i])
            dCdW[-i] = np.dot(activations[-i-1].T, delta) / m
            dCdb[-i] = np.sum(1 * delta, axis=0) / m

        return dCdW, dCdb

    def gradient_descent(self, inputs, outputs, eta):
        """
        Updates weights and biases using gradient descent.

        Parameters
        ----------
        inputs: numpy array
            Input values
        outputs: numpy array
            Outputs values
        eta: float
            Learning rate to use for gradient descent
        """
        gradient_weights, gradient_biases = self.back_propagation(inputs, outputs)
        for W, b, dW, db in zip(self.weights, self.biases, gradient_weights, gradient_biases):
            W += -eta * dW
            b += -eta * db

    def train(self, callback=None):
        """
        Train the neural network.

        Parameters
        ----------
        callback: function
            Function to run after every fifth epoch
        """
        loss = np.mean(self.cost_function(self.feed_forward()[-1], self.outputs), axis=0)[0]
        print(f"Epoch 0, loss: {loss}")
        for epoch in range(1, self.epochs + 1):
            # Combine input and output data
            data = np.concatenate((self.inputs, self.outputs), axis=1)
            # Randomly shuffle data
            np.random.shuffle(data)
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]
                inputs, outputs = batch[:, : -self.layers[-1]], batch[:, -self.layers[-1]:]
                self.gradient_descent(inputs, outputs, self.eta)

            # Calculate loss
            loss = np.mean(self.cost_function(self.feed_forward()[-1], self.outputs), axis=0)[0]

            if epoch % 100 == 0:
                print(f"Epoch: {epoch} - loss: {loss}")

            if callback is not None and self.epochs == epoch: #epoch % 5 == 0:
                callback(self, epoch, loss)

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
        for W, b, activation_type in zip(self.weights, self.biases, self.activations):
            activations.append(self.activation_function(np.dot(activations[-1], W) + b, activation_type))
        return activations[-1]
