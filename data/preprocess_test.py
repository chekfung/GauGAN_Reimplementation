from skimage.io import imread
import numpy as np 
import sys
import matplotlib.pyplot as plt
#import convertMATIndexToCSV
from convertMATIndexToCSV import ADEIndex


matindex = ADEIndex()

filename = "/Users/ChekF/Desktop/College/sophomore_year/cs1430/James_TompGAN/data/ADE20K_2016_07_26/images/training/l/lean-to/ADE_train_00010929_seg.png"


meow = imread(filename)

# R g and b channels from the file
r = meow[:,:,0]
g = meow[:,:,1]
b = meow[:,:,2]

r = r.astype(np.uint16)
g = g.astype(np.uint16)

obj_mask =  np.floor((r / 10)) * 256 + g
object_set = set()

# For each pixel in the object index mask: 
for i in range(obj_mask.shape[0]):
    for j in range(obj_mask.shape[1]):
        
        # If the index is 0, increase by 1
        if obj_mask[i,j] == 0:
            obj_mask[i,j] = 1

        # # -1 in indexing from matindex because MATLAB starts at 1 and not 0
        object_name = matindex.object_name_list['objectnames'].loc[int(obj_mask[i,j] - 1)]
        object_set.add(object_name)


print(object_set)
print(obj_mask)





# grass 
# sea
# field
# clouds
# fog
# hill
# mountain
# river
# rock
# snow
# stone
# water
# dirt
# gravel
# tree
# sand
# earth



