## MNIST classification with Pytorch
The [notebook](https://github.com/namanphy/EVA5/blob/main/S4/eva_session4_architectural_basics.ipynb) contains the MNIST classification with Pytorch as per EVA session 4 
with a target of achieving 99.4% accuracy.

### Aim
Following targets must be achieved in the notebook.
1. 99.4% validation accuracy
2. Less than 20k Parameters
3. Less than 20 Epochs
4. No fully connected layer

### Process
Here, We have

- Input image is of size - `1x28x28`.
- Output is of size - `1x10`.

- We have used 4 convulation layers, along with 2 Batch normalisation and 2 Maxpool layers. At last the architecture is kept such the number of channels are converted to 10 to 
match the output shape. Lastly, Global Average Pooling is applied on the final output to make the shape from `10x3x3` to `10x1x1` which is later reshaped and converted to 
log likelihood.
**Hence no fully connected layer is used.**

  Have a look at the model summary bellow.

  ![architecture](https://github.com/namanphy/EVA5/blob/main/S4/arch.png)


- The total number of trainable parameters are = **19,450**.

- The model is then trained for 20 epochs. The learning rate is selected as `0.01` for first 12 epochs and for last 8 epochs it was changed to `0.009`. This was selected after
conduting some iterations over the model.

- At the 18th epoch we were able to reach **98.41% accuracy**.


### Conclusion
1. The use of batch normalisation helps the model to achieve the desired accuracy. 
2. The model is designed keeping in mind the global **Receptive Fields** at each layer. The target is to reach at least `28x28` RF. 
3. The number of channels were small but selected in a non-decreasing order.

