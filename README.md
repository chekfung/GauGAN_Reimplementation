# James_TompGAN
Implementation of the GauGAN Nvidia research paper in Tensorflow rather than Pytorch.

Note: This repository will not contain the ADE20k dataset due to its size. The
dataset (ADE20K_2016_07_26/) should be in the data/ directory.

The original ADE20K_2016_07_26/ dataset is prepared for the network by running
get_landscape_img.py, which selects images and corresponding segmentation maps
from the dataset as follows:
- **explicit_cv_landscapes_final_project.txt** lists explicit scene categories
  to include 
- **objects_we_want.txt** lists words and phrases (each phrase on a separate
  line) that describe objects, and all images that are known to contain
  at least one of the listed objects are included


Changes Made: 
- Shrank image sizes to 40x30
- Reduced the number of spade layers in Generator
- Reduced z-dimension to 64
- 
