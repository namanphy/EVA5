## Dataset

This directory contains the HardhatVestMaskBoots dataset used for training and 
testing the model. The dataset can be downloaded from this link.

For custom dataset:
 
Update the files <>.names, <>.data, <>.shapes, train.txt and test.txt accordingly.

The following tool is used to annotate the dataset:
[https://github.com/miki998/YoloV3_Annotation_Tool](https://github.com/miki998/YoloV3_Annotation_Tool)

## File Structure

- `images` : all the images used
- `labels` : labels of the images (as got from annotation tool)
- `test` : txt and shape files containing test images
- `train` : txt and shape files containing train images
- `<>.names` : class names seperated by '\n'
- `<>.data` : entry point for the model to the dataset