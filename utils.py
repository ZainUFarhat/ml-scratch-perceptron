# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Calculate accuracy - out of 100 examples, what percentage does our model get right?
def accuracy_fn(y_true, y_pred):
  """
  calculates the accuracies of a given prediction

  Parameters:

    y_true: the true labels
    y_pred: our predicted labels
  
  Returns:

    accuracy
  """
  
  # find the number of correct predictions  
  correct = np.equal(y_true, y_pred).sum()
  # calculate the accuracy
  acc = (correct/len(y_pred))*100
  # return the accuracy
  return round(acc, 2) 

# unit step func
def unit_step_func(x):
  
  """
  Description:
    Unit step activation function for our perceptron

  Parameters:
    x: linear output of perceptron

  Returns:
    predicted_class
  """

  # find predicted class
  predicted_class = np.where(x > 0, 1, 0)

  # return
  return predicted_class

# scatter plot of given data
def scatter_plot(X, y, title, x_label, y_label, class_names, savepath):

    """
    Description:
        Plots a scatterplot based on X & y data provided

    Parameters:
        X: x-axis datapoints
        y: y-axis datapoints
        title: tite of plot
        x_label: label for x axis
        y_label: label for y axis
        class_names: names of our target classes
        savepath: path to save our scatterplot to

    Returns:
        None
    """

    # intialize figure
    plt.figure(figsize = (7, 7))

    # set background color to lavender
    ax = plt.axes()
    ax.set_facecolor("lavender")

    # find features corresponding to class labels
    class_0, class_1 = X[y == 0], X[y == 1]

    # scatter plots of class features against themselves
    plt.scatter(class_0[:, 0], class_0[:, 1], label = class_names[0], c = 'r')
    plt.scatter(class_1[:, 0], class_1[:, 1], label = class_names[1], c = 'b')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend()
    plt.savefig(savepath)

    # return
    return None

# visualize decision boundary
def visualize_decision_boundary(X, y, model, title, x_label, y_label, class_names, resolution, savepath):
   
   """
   Description:
        Plot the decision doundary of a logistic regression model

   Parameters:
        X: features
        y: targets
        model: model to fit our meshgrid on
        title: title for our plot
        x_label: x label of our plot
        y_label: y label of our plot
        class_names: names of our classes
        resolution: resolution of grid for plotting the decision boundary
        savepath: path to save our decision plot boundary to

   Returns:
        None
   """

   # print(X.shape, y.shape)

   # initialize figure
   plt.figure(figsize = (7, 7))

   # scatter plot based on class labels
   # find features correspaonding to class labels
   class_0, class_1 = X[y == 0], X[y == 1]
   
   # scatter plots of class features against themselves
   plt.scatter(class_0[:, 0], class_0[:, 1], label = class_names[0], c = 'r')
   plt.scatter(class_1[:, 0], class_1[:, 1], label = class_names[1], c = 'b')
   plt.title(title)
   plt.xlabel(x_label)
   plt.ylabel(y_label)
   plt.grid()
   plt.legend()
   
   # get the min and max limits for our x-axis and y-axis
   # here the two features chosen for X_train_reduced will compose the x-axis and y-axis respectively
   x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   
   # we will create a meshgrid based on the x-axis and y-axis range
   # we will take resolution steps between min and max and we are aiming to classify all of these into our given labels
   xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

   # predict the labels for all points on the created meshgrid
   predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
   Z = np.array(predictions)
   Z = Z.reshape(xx.shape)

   # Original blue and red colors
   original_blue = (0.0, 0.0, 1.0)  # RGB values for blue
   original_red = (1.0, 0.0, 0.0)   # RGB values for red

   # Adjust the brightness (increase the value for a brighter color)
   brighter_factor = 1.5
   brighter_blue = tuple(min(1.0, c * brighter_factor) for c in original_blue)
   brighter_red = tuple(min(1.0, c * brighter_factor) for c in original_red)

   # Create a custom colormap with the brighter colors
   cmap_brighter = ListedColormap([brighter_red, brighter_blue])

   # Plot the decision boundary
   plt.pcolormesh(xx, yy, Z, cmap = cmap_brighter, alpha = 0.3)

   plt.xlim(-0.05, 1.05)
   plt.ylim(-0.05, 1.05)

   # save fig
   plt.savefig(savepath)

   # return
   return None