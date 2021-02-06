## CIFAR-10 classification - ResNet18 & Albumentations & GradCam

This [repo](https://github.com/namanphy/EVA5/blob/main/S9) contains the code for 
CIFAR-10 classification with Pytorch using RESNET-18 model.

### Aim
The goal here is to 
- use ResNet18 architecture
- Achieve more than 87% accuracy
- Use **Albumentation** library for transformations - This is a beautiful library for 
various low latency transformations and augmentations.
- Implementing and using **Gradcam**


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
- Number of parameters - 11,173,962
- We have used ResNet-18 architecture having 18 Regular Resnet blocks with Residual connections.
- Fully connected layer is used at last to make the final output to `10x1x1`.


### Results

#### Accuracy curve
![accuracy](https://github.com/namanphy/EVA5/blob/main/S9/images/accuracy.png)

#### Loss curve
![accuracy](https://github.com/namanphy/EVA5/blob/main/S9/images/loss.png)

#### Misclassified Images
Showing 25 random misclassified images.
![misclassified images](https://github.com/namanphy/EVA5/blob/main/S9/images/incorrect_predictions.png)

### Gradcam

The gradcam is applied on a image belonging to one of the classes of 
CIFAR10 dataset. It shows what the model is seeing at the last layer 
before the prediction.

This is the normal image for an aeroplane.

![airplane](https://github.com/namanphy/EVA5/blob/main/S9/images/airplane.png)



The image after gradcam is applied. It shows what the model is seeing 
at the last layer before the prediction. Red areas show higher activation 
and vice-versa.

![grad airplane](https://github.com/namanphy/EVA5/blob/main/S9/images/gradcam-ResNet-layer4.1.conv2-airplane.png)


#### Quick Run: Go to Google Colab
Select Python 3 as the runtime type and GPU as the hardware accelerator and run the notebook - `cifar10_resnet18_main.ipynb`.
Make sure to install Pytorch.
