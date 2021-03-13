# TinyImageNet and Yolo Anchor Boxes

## Part A - Object classification on Tiny ImageNet


This [repo](https://github.com/namanphy/EVA5/blob/main/S12) contains the object
classification on Tiny ImageNet dataset with ResNet18. The dataset has 200 classes.


### Parameters and Hyperparameters
- Kernel Size: 3x3
- Input image is of size - `3x64x64`.
- Output is of size - `1x200`.

Training
- Batch Size: 128
- Learning Rate: 0.01
- Epochs: 40

Optimizer and Loss
- Optimizer: SGD
- Loss Function: Cross entropy

Reduce LR on Plateau
- patience: 2
- factor: 0.1
- min-lr: 0.000001

### Results

#### Accuracy curve
Following is the accuracy curve during training and testing 
![accuracy](https://github.com/namanphy/EVA5/blob/main/S12/images/accuracy.png)

--------

## Part B - Identifying Clusters - Yolo bounding Box



#### Quick Run: Go to Google Colab
Select Python 3 as the runtime type and GPU as the hardware accelerator and run the notebook - `main_S10.ipynb`.
Make sure to install Pytorch.
