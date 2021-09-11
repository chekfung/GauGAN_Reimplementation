# GauGAN Reimplementation

## Implementation of [NVIDIA's SPADE Normalization Layer](https://nvlabs.github.io/SPADE/) for the Pix2Pix GAN

NVIDIA SPADE is a normalization layer that augments the Pix2Pix GAN architecture for semantic
image synthesis. The NVIDIA paper is implemented using PyTorch, while this project
is written with TensorFlow.

Our implementation is trained and tested on a subset of the [MIT ADE20K
scene segmentation dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/),
 to decrease the computational demands of training.
Our subset of the dataset selected images with "landscape-like" objects, such as
beaches and trees, and did not include images with non-landscape-like objects, 
such as couches.

In order to train our models on the computing resources available to us, we
made other modifications to the original NVIDIA implementation, which are listed
at the bottom of this file.

Note: This repository will not contain the ADE20k dataset due to its size. The
dataset (ADE20K_2016_07_26/) should be in the data/ directory.

## Preparing the Dataset

The original ADE20K_2016_07_26/ dataset is prepared for the network by running
get_landscape_img.py, which selects images and corresponding segmentation maps
from the dataset as follows:
- **explicit_cv_landscapes_final_project.txt** lists explicit scene categories
  to include 
- **objects_we_want.txt** lists words and phrases (each phrase on a separate
  line) that describe objects, and all images that are known to contain
  at least one of the listed objects are included

## Changes from Original Paper Implementation: 
- Shrank image sizes to 128x96
- Reduced the number of upsampling layers in the generator from 7 to 5 
- Reduced z-dimension to 64
- We specifically do not include the encoder in our generator. It is in the code, but not used.
