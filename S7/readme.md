## CIFAR-10 classification with Pytorch

This [repo](https://github.com/namanphy/EVA5/blob/main/S7) contains the code for 
CIFAR-10 classification with Pytorch.

### Aim
The goal here is to 
- C1C2C3C4 architecture
- Total Receptive Field > 44
- To use **Depthwise Separable Convolution** and **Dilated Convolution**
- Achieve 80% accuracy with total params less than 1M


### Parameters and Hyperparameters
- Kernel Size: 3x3
- Optimizer: SGD
- Loss Function: Cross entropy
- Dropout Rate: 0.01
- Batch Size: 64
- Learning Rate: 0.01
- Dilation factor: 4


### Process
Here, We have
<!-- ![architecture](https://github.com/namanphy/EVA5/blob/main/S4/mnist-1.png) -->

- Input image is of size - `3x32x32`.
- Output is of size - `1x10`.
- We have used 4 convolution blocks, along with 3 Transition blocks having Maxpool layers.
- Fully connected layer is used at last to make the final output to `10x1x1`.
- The total number of trainable parameters are = **520,682**.

### Results

##### Accuracy curve
![accuracy](https://github.com/namanphy/EVA5/blob/main/S7/images/accuracy.png)

##### Loss curve
![accuracy](https://github.com/namanphy/EVA5/blob/main/S7/images/loss.png)

##### Misclassified Images
![accuracy](https://github.com/namanphy/EVA5/blob/main/S7/images/misclassification.png)


#### On Google Colab
Select Python 3 as the runtime type and GPU as the hardware accelerator and run the notebook - `cifar10-main.ipynb`.
Make sure to install Pytorch.
