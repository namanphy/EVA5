## CIFAR-10 classification - Finding correct LR

This [repo](https://github.com/namanphy/EVA5/blob/main/S10) contains the code for 
CIFAR-10 classification with Pytorch using RESNET-18 model.


The goal here is to find the best learning rate using LR finder. The following 
is the graph of training loss vs learning rate increased exponentially.

![loss_vs_lr](https://github.com/namanphy/EVA5/blob/main/S10/images/loss_vs_lr.png)

Seeing from the above graph - 0.001 to 0.006 is where loss is decreasing steeply
and thus choosing 0.005 as starting LR.


### Parameters and Hyperparameters
- Kernel Size: 3x3
- Input image is of size - `3x32x32`.
- Output is of size - `1x10`.

Training
- Batch Size: 64
- Learning Rate: 0.005
- Epochs: 50

Optimizer and Loss
- Optimizer: SGD
- Loss Function: Cross entropy

LR Finder
- factor: 0.5
- patience: 2

Gradcam
- Layer: layer4.1.conv2


### Results

#### Accuracy curve
Following is the accuracy curve during training and testing 
![accuracy](https://github.com/namanphy/EVA5/blob/main/S10/images/accuracy.png)

--------

#### Misclassified Images
Showing 25 random misclassified images.

![misclassified images](https://github.com/namanphy/EVA5/blob/main/S10/images/incorrect_predictions.png)

---------

#### Gradcam for misclassified images
Here are some of the misclassified images along 

![grad images](https://github.com/namanphy/EVA5/blob/main/S10/images/incorrect_predictions_gradcam_sample.png)


#### Quick Run: Go to Google Colab
Select Python 3 as the runtime type and GPU as the hardware accelerator and run the notebook - `main_S10.ipynb`.
Make sure to install Pytorch.
