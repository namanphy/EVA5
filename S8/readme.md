## CIFAR-10 classification - using ResNet-18

This [repo](https://github.com/namanphy/EVA5/blob/main/S8) contains the code for 
CIFAR-10 classification with Pytorch using RESNET-18 model.

### Aim
The goal here is to 
- use ResNet18 architecture
- Achieve more than 85% accuracy


### Parameters and Hyperparameters
- Kernel Size: 3x3
- Optimizer: SGD
- Loss Function: Cross entropy
- Batch Size: 64
- Learning Rate: 0.01


### Process
Here, We have
<!-- ![architecture](https://github.com/namanphy/EVA5/blob/main/S4/mnist-1.png) -->

- Input image is of size - `3x32x32`.
- Output is of size - `1x10`.
- We have used ResNet-18 architecture having 18 Regular Resnet blocks with Residual connections.
- Fully connected layer is used at last to make the final output to `10x1x1`.


### Results

##### Accuracy curve
![accuracy](https://github.com/namanphy/EVA5/blob/main/S8/images/accuracy.png)

##### Loss curve
![accuracy](https://github.com/namanphy/EVA5/blob/main/S8/images/loss.png)

#### On Google Colab
Select Python 3 as the runtime type and GPU as the hardware accelerator and run the notebook - `cifar10_resnet18_main.ipynb`.
Make sure to install Pytorch.
