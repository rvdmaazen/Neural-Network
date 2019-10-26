from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import sklearn.datasets


def plot_decision_boundary(network, epoch, loss):
    # Create grid points
    xx, yy = np.meshgrid(np.arange(-5, 5, 0.01), np.arange(-5, 5, 0.01))

    # Get decision boundary
    probs = network.predict(np.c_[xx.ravel(), yy.ravel()])
    probs = probs.reshape(xx.shape)

    # Plot decision boundary gradient
    plt.contourf(xx, yy, probs, cmap="Spectral_r", alpha=0.6)
    # Plot decision boundary
    plt.contour(xx, yy, probs, levels=[0.5], cmap="Greys", vmin=0, vmax=0.6)

    # Get training data
    x, y = network.inputs, network.outputs
    # Plot training data
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), cmap="seismic", edgecolors="black")

    # Set title
    plt.suptitle(f"Epoch {epoch}, loss: {loss:.10f}", fontsize=16)
    # Set subtitle
    plt.title(
        f"Learning rate: {network.eta}, batch size: {network.batch_size},\nlayers: {network.layers}, activations: {network.activation_types}", fontsize=12
    )
    # Lower top of plot to make room for multi-line subtitle
    plt.subplots_adjust(top=0.85)
    # Set axis labels
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    # Set axis limits
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    # Save and close plot
    plt.savefig(f"visualizations/test_{epoch:05d}.png")
    plt.close()


if __name__ == "__main__":

    # Set Computer Modern font
    rc("font", **{"family": "serif", "serif": ["CMU Serif"]})
    rcParams["mathtext.fontset"] = "cm"

    # Seed for data generation
    SEED = 123
    # Number of observations
    N = 200
    # Amount of noise in data
    NOISE = 0.1

    # Blobs (make_blobs)
    x, y = sklearn.datasets.make_blobs(n_samples=N, centers=[(-1.5, -1.5), (1.5, 1.5)], random_state=SEED)
    nn = NeuralNetwork([2, 10, 10, 1], x, y, 10000, eta=0.03, activations=["relu", "relu", "sigmoid"], loss="mean_squared_error")
    #nn.train(callback=plot_decision_boundary)

    # Circles (make_circles)
    x, y = sklearn.datasets.make_circles(n_samples=N, noise=NOISE, factor=0.5)
    nn = NeuralNetwork([2, 10, 10, 1], 3 * x, y, 10000, eta=0.03, activations=["relu", "relu", "sigmoid"], loss="mean_squared_error")
    #nn.train(callback=plot_decision_boundary)

    # make_classification
    x, y = sklearn.datasets.make_classification(
        n_samples=N, n_features=2, n_informative=2, n_redundant=0
    )
    nn = NeuralNetwork([2, 10, 10, 1], x, y, 10000, eta=0.03, activations=["relu", "relu", "sigmoid"], loss="mean_squared_error")
    #nn.train(callback=plot_decision_boundary)

    # Moons (make_moons)
    x, y = sklearn.datasets.make_moons(n_samples=N, noise=0.1, random_state=SEED)
    nn = NeuralNetwork([2, 10, 10, 1], 2 * x, y, 10000, eta=0.03, activations=["relu", "relu", "sigmoid"], loss="mean_squared_error")
    nn.train(callback=plot_decision_boundary)
