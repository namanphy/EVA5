# MNIST classification with Pytorch - using BN and GBN

The notebooks contains the MNIST classification with Pytorch with given regularisation techniques.

## AIM
The goal of this assignment is to apply regularization techniques on the mnist model and observe the changes in 
validation loss and accuracy obtained during model training in the following scenarios:

- Batch Normalisation with L1 regularization
- Batch Normalisation with L2 regularization
- Batch Normalisation with L1 and L2 regularization
- Ghost Batch Normalisation
- Ghost Batch Normalisation with L1 and L2 regularization



## Parameters and Hyperparameters
- Kernel Size: 3x3
- Optimizer: SGD
- Loss Function: Negative Log Likelihood
- Dropout Rate: 0.01
- Batch Size: 64(for BN models) and 128(for GBN models)
- Learning Rate: 0.01
- L1 Factor: 0.001
- L2 Factor: 0.0004


## Results

#### Accuracy curve
![accuracy](https://github.com/namanphy/EVA5/blob/main/S6/images/accuracy.png)

#### Loss curve
![accuracy](https://github.com/namanphy/EVA5/blob/main/S6/images/loss.png)

#### Misclassified Images
![accuracy](https://github.com/namanphy/EVA5/blob/main/S6/images/incorrect_predictions.png)


## Start Guide
Download the above notebook and run accrdingly.

#### On Google Colab
Select Python 3 as the runtime type and GPU as the harware accelerator.

