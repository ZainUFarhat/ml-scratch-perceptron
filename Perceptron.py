# numpy
import numpy as np

# utils
from utils import *

# perceptron
class Perceptron():

    """
    Description:
        My from scratch implementation of the Perceptron Algorithm 
    """

    # constructor
    def __init__(self, epochs, lr):

        """
        Description:
            Constructor for our Perceptron class

        Parameters:
            epochs: number of iterations to train our perceptron to
            lr: learning rate
        
        Returns:
            None
        """

        self.epochs = epochs
        self.lr = lr
        self.activation_func = unit_step_func
        self.weights = []
        self.bias = 0
    
    # fit
    def fit(self, X, y):

        """
        Description:
            Fits our perceptron
        
        Parameters:
            X: train features
            y: train labels
        
        Returns:
            None
        """

        # fetch number of features
        N, num_features = X.shape

        # intialize weights to zeros array of shape 1 x num_features
        self.weights = np.zeros(num_features)

        # make sure classes are only 0 & 1, i.e. not -1 and 1
        y = np.where(y > 0, 1, 0)

        # iterate over dataset and learn representable weights
        for _ in range(self.epochs):

            for (i, x_i) in enumerate(X):

                # compute linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                # find predicted class
                y_predicted = self.activation_func(linear_output)

                # perform updates for weights and bias
                update = self.lr * (y[i] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    # predict
    def predict(self, X):

        """
        Description:
            Predicts on our fitted perceptron model

        Parameters:
            X: test features
        
        Returns:
            predictions
        """

        # compute linear output
        linear_output = np.dot(X, self.weights) + self.bias
        # find predictions
        predictions = self.activation_func(linear_output)

        # return 
        return predictions