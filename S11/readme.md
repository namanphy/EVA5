# Super Convergence

## Part A - Triangular schedule curve
Here in this [notebook](https://github.com/namanphy/EVA5/blob/main/S11/cyclicLR_S11.ipynb) has
the code to make LR test - triangular schedule curve. Follow the steps to make a curve like follwing :

![LRTest](https://github.com/namanphy/EVA5/blob/main/S11/images/clr_plot.png)

---------

## Part B - Training ResNet18 on CIFAR10 to 90% in 24 epochs

Here in this [notebook](https://github.com/namanphy/EVA5/blob/main/S11/main_S11.ipynb) has the required 
code to perform the following test and steps for training of ResNet18 on CIFAR10.

It is using a custom package present in this repo.


### LR Finder Test

This test let the model to run for several epochs while letting the learning rate increase linearly between low
and high LR values.

Max epochs = 10
LR MIN =  0.0001
LR MAX = 0.05


![LRFinder](https://github.com/namanphy/EVA5/blob/main/S11/images/lr_range_test-linear.png)

Choosing the max LR from above as - **0.02**


### Parameters and Hyperparameters

Training
- Batch Size: 512
- Epochs: 24

Optimizer and Loss
- Optimizer: SGD
- Loss Function: Cross entropy

Image Augmentations
- Padding: 4
- RandomCrop: (32,32)
- Horizontal Flip: 20%
- CutOut: (8, 8) - 50%


One Cycle LR
- Max at Epoch = 5
- max-lr: 0.002
- NO Annihilation

### Results

#### Accuracy curve
Following is the accuracy curve during training and testing 
![accuracy](https://github.com/namanphy/EVA5/blob/main/S11/images/accuracy.png)


We have able to achieve **89.51 %** in 24 epochs.

## Quick Run: Go to Google Colab
Select Python 3 as the runtime type and GPU as the hardware accelerator and run the 
notebook - `main_S12.ipynb`.
Make sure to install Pytorch.
