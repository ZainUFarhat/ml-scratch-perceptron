# ml-scratch-perceptron
Perceptron Algorithm

## **Description**
The following is my from scratch implementation of the Perceptron algorithm.

### **Dataset**

I tested the performance of my model on three datasets: \
\
    &emsp;1. Sklearn Blobs Dataset \
    &emsp;2. Breast Cancer Dataset \
    &emsp;3. Diabetes Dataset

### **Walkthrough**

**1.** Need the following packages installed: sklearn, numpy, and matplotlib.

**2.** Once you made sure all these libraries are installed, evrything is simple, just head to main.py and execute it.

**3.** Since code is modular, main.py can easily: \
\
    &emsp;**i.** Load the three datasets \
    &emsp;**ii.** Split data into train and test sets \
    &emsp;**iii.** Build a perceptron classifier \
    &emsp;**iv.** Fit the perceptron classifier \
    &emsp;**v.** Predict on the test set \
    &emsp;**vi.** Plot scatter plots and decision boundaries for each dataset.

### **Results**

For each dataset I will share the test accuracy and show the decision boundary predictions.

**1.** Sklearn Blobs Dataset:

- Numerical Result:
     - Accuracy = 100.0%

- See visualization below:

 ![alt text](https://github.com/ZainUFarhat/ml-scratch-perceptron/blob/main/plots/blobs/blobs_decision_boundary.png?raw=true) 

**2.** Breast Cancer Dataset:

- Numerical Result:
     - Accuracy = 95.61%

- See visualization below:

 ![alt text](https://github.com/ZainUFarhat/ml-scratch-perceptron/blob/main/plots/bc/bc_decision_boundary.png?raw=true)

**2.** Diabetes Dataset:

- Numerical Result:
     - Accuracy = 70.79%

- See visualization below:

 ![alt text](https://github.com/ZainUFarhat/ml-scratch-perceptron/blob/main/plots/db/db_decision_boundary.png?raw=true)