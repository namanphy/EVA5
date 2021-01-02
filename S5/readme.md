# MNIST classification with Pytorch
The notebooks contains the MNIST classification with Pytorch with each ipynb explaining steps taken to approach the 
given problem.


### Aim
Following targets must be achieved in the notebook.
 - 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
 - Less than or equal to 15 Epochs
 - Less than 10000 Parameters

### Result
The final result is : 
 - 99.46% test accuracy achieved. (with consistently above 99.40% for last 8 epochs)
 - 15 epochs were used.
 - 6398 parameters were used in the model.
 - Batch size of 64 is used.
 
 
## Step 1 - [notebook](https://github.com/namanphy/EVA5/blob/main/S5/eva_session5%20-%20iter%201.ipynb)

Target
1. Set up the imports, Dataloader, and basic train and test pipeline.
2. Visualise the dataset.
3. Get the basic model up and running and identify the next step.

Results
1. parameters = 24,272
2. Train Accuracy = 99.63
3. Test Accuracy = 99.08

Analysis
1. Good training accuracy to start with only 24,000 parameters and no fancy stuff. The model architecture is somewhat good.
2. Overfitting - It is clear that our basic model is overfitting.
3. Also evident from the graphs that `test loss` is not consistently decreasing.
4. Number of model parameters is actually more than the required target.


## Step 2 - [notebook](https://github.com/namanphy/EVA5/blob/main/S5/eva_session5%20-%20iter%202.ipynb)

Target
1. To reduce the number of parameters.

Results
1. parameters = 8,734
2. Best Train Accuracy = 99.34
3. Best Test Accuracy = 99.00

Analysis
1. Good training accuracy with reduced number of parameters.
2. Overfitting - The model is still overfitting.
3. Also the best training and test accuracies are reduced.
4. Number of model parameters is under required target.


## Step 3 - [notebook](https://github.com/namanphy/EVA5/blob/main/S5/eva_session5%20-%20iter%203.ipynb)

Target
1. And reducing the last layer with 1x1 layer - To reduce the number of parameters.
2. Using GAP after 1x1 layer.
3. Removed Padding.

Results
1. parameters = 4,894
2. Best Train Accuracy = 97.84
3. Best Test Accuracy = 97.57

Analysis
1. The model is good and can be better if pushed further.
2. Overfitting - The model is little overfitting at last few epochs.
3. The number of parameters are further reduced and thus decreased the capacity of the model.


## Step 4 - [notebook](https://github.com/namanphy/EVA5/blob/main/S5/eva_session5%20-%20iter%204.ipynb)

Target
1. Add BatchNormalisation to increase the model efficiency.
2. Added more capacity at the end. Just before the 1x1 layer.
3. Reduced the batch size - To help the model escape any local minima.

Results
1. parameters = 6,398
2. Best Train Accuracy = 99.58
3. Best Test Accuracy = 99.25

Analysis
1. Increased training accuracy after using Batch Normalisation - and it is well above target.
2. Overfitting - The model is still overfitting - After 7th epoch it started shoowing signs of overfitting.
3. The model is becomming good and can be better if pushed further. But it is not robust as the test accuracy keeps on fluctuating.


## Step 5 - [notebook](https://github.com/namanphy/EVA5/blob/main/S5/eva_session5%20-%20iter%205.ipynb)

Target
1. Apply LR Scheduler
In previous iteration, the model accuracy was decreasing after 5th epoch so LR step is set to 5.
2. Used rotation in transformation.

Results
1. parameters = 6,398
2. Best Train Accuracy = 99.28
3. Best Test Accuracy = 99.44

Analysis
1. Good Model! The model has crossed the target accuracy. Less fluctuation in test accuracy.

2. The model is underfitting which is expected.

3. The model performed well in last 7-8 epochs maintaining a greater then 99.41% for 4-5 epochs. Still not showing complete consistency in accuracy.

4. The model is capable of pushing further.


## Step 6 - [notebook](https://github.com/namanphy/EVA5/blob/main/S5/eva_session5%20-%20iter%206.ipynb)

Target
1. Using dropout to avoid any overfitting and complement the added transformations in train data
2. dropout value = 0.02 (2%)

Results
1. parameters = 6,398
2. Best Train Accuracy = 99.32
3. Best Test Accuracy = 99.46

Analysis
1. Great Model! The model performed well in last 8 epochs maintaining a greater then 99.40% for all the time.

2. Dropout helped the model to achieve steadiness.

3. The model is becomming good and can be better if pushed further. Less fluctuation in test accuracy.

