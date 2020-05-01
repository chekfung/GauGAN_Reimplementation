import numpy as np
import os 
import cv2
from copy import deepcopy
from tqdm import tqdm

def extract_labels(data_dir): 
	with open(data_dir, "r") as file: 
		label_colors = file.read().split("\n")[:-1]
		label_colors = [x.split("\t") for x in label_colors]
		colors = [x[0] for x in label_colors]
		colors = [x.split(" ") for x in colors]

	for i,color in enumerate(colors):
		colors[i] = [int(x) for x in color]
	
	return np.array(colors)