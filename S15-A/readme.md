# Generating Depth and Planer region images

This folder contains the links to the Inverse Depth Maps using MiDaS (https://github.com/intel-isl/MiDaS) and the PlaneR-CNN (https://github.com/NVlabs/planercnn) that detects arbitrary number of planes and reconstructs piecewise planar surfaces from RGB images.

The source images for this are picked from the Dataset that contained images collected for YOLO V3 object detection (Exists in the S13 folder in this repo, can also be viewed from here - https://drive.google.com/drive/folders/1-GVUStQkkwA8hx8lFZe9SyY-yvr7-YvT?usp=sharing). This Dataset had around 3500 images with 4 classes - Hardhat, Vest, Mask, Boots - all of which are safety gear. 

# MiDaS

The Dataset was run through the "MiDaS v2.1 large" model and the inverse depthmaps for each image were generated.

## Original Image:

![C20](https://user-images.githubusercontent.com/65642947/113488234-58e33a00-94da-11eb-81c8-f67f8960bdef.jpg)


## Depth Map obtained through MiDaS:

![C20](https://user-images.githubusercontent.com/65642947/113488249-67315600-94da-11eb-8cda-11bc267ac58f.png)


## Google Drive Link of the entire Dataset run on MiDaS:

https://drive.google.com/drive/folders/1MclNOIaaMZ9SmeHG7hoQ3KvF8IHWTJPw?usp=sharing


# PlaneR-CNN

The Dataset was run through the "planercnn_normal_warping_refine" model that performed image segmentation on a planar level.

## Original Image:

![413_image_0](https://user-images.githubusercontent.com/65642947/113488415-9f856400-94db-11eb-84fc-0f281c7f09bd.png)


## Planar Segmentation obtained through PlaneR-CNN

![413_segmentation_0_final](https://user-images.githubusercontent.com/65642947/113488437-c3e14080-94db-11eb-8c65-475167ec090b.png)


## Google Drive Link of the entire Dataset run on PlaneR-CNN:

https://drive.google.com/drive/folders/1iNK5UxVwyCfN4t81OTrMtJJZUVXGg0Q_?usp=sharing


### Contributors

1. Abhijeet
2. Naman Bhardwaj
3. Rangasai K R
4. Urmila



