# Neural Network Using NumPy
A flexible implementation of a fully connected neural network in Python using NumPy, allowing multiple different activation functions, loss functions, and network structures. Includes visualization of neural network learning process.

## Installation
1. Clone this repository

    `git clone github.com/rvdmaazen/Neural-Network.git`

2. Change directory to the cloned repository

    `cd Neural-Network-NumPy`

3. Install the necessary Python packages

    `pip install -r requirements.txt`

## Visualizations
Visualizing the learning process of a neural network was done using a callback to the `plot_decision_boundary` defined function in `visualizations.py`. This function plots the decision boundary of the neural network every 5 epochs using `matplotlib` and saves the plot as an image. These images are then combined into a gif using [ImageMagick](https://imagemagick.org/). Training data was created using the `datasets` package from `scikit-learn`.

### Blobs (make_blobs)
![](visualizations/make_blobs.gif)

### General classification (make_classification)
![](visualizations/make_classification.gif)

### Circles (make_circles)
![](visualizations/make_circles.gif)

### Moons (make_moons)
![](visualizations/make_moons.gif)

### Effect of activation functions
For this specific dataset, using a sigmoid activation function (top left) for the layers does not allow the network to correctly converge. After 10000 epochs the decision boundary is straight and does not follow the distribution of the data at all, resulting in a loss of 0.0895. 

Using the tanh (hyperbolic tangent) activation function (top right) provides a substantial improvement over the sigmoid activation function, the decision boundary follows the data distribution much better resulting in a much lower loss of 0.0135. 

The ReLu (rectified linear unit) activation function (bottom left) shows similar performance as the tanh activation function. A significant improvement is seen compared to the sigmoid activation function, however the loss is slightly higher when compared to the tanh activation function (0.0172 vs 0.0135). 

Finally, the ELU (exponential linear unit) activation function (bottom right) is able to follow the distribution of the data reasonably well but not quite as well as the tanh and ReLU, resulting in a loss of 0.0386. Overall, using the tanh activation function for the hidden layers seems to perform the best on this dataset.

![](visualizations/activation_functions.gif)

### Effect of neural network architecture
Deeper and wider networks have higher flexibility allowing it to better fit to the signal in the data, in turn decreasing the bias and increasing test accuracy. However, this will also mean that the network will try to fit to noise in the data, which lease to a higher variance (in accordance to bias-variance tradeoff) and a lower test/validation accuracy (also known as overfitting). [Early stopping](https://en.wikipedia.org/wiki/Early_stopping) and [dropout layers](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) in a neural network are examples of solutions that aim to help minimize the problem of overfitting. 

For this example, one can see that the both the deeper and wider networks converge quicker than the baseline (2 hidden layers with 10 neurons, top left). The deeper network (4 hidden layers with 10 neurons, top right) converges quicker than the wide network (2 hidden layers with 25 neurons, bottom left). The deep and wide network (4 hidden layers with 25 neurons, bottom right) converges the quickest.

![](visualizations/network_architecture.gif)