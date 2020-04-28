from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='GAUGAN')

parser.add_argument('--printCSVs', type=str, default='N',
					help='Whether to print CSV files from MATLAB index')

class ADEIndex():

  # This script should be run from within the project's James_TompGAN/data/ folder

  data_dir = pjoin('ADE20K_2016_07_26')

  mat_fname = pjoin(data_dir, 'index_ade20k.mat')

  mat_contents = sio.loadmat(mat_fname)

  matindex = mat_contents['index'][0,0]

  # print("Index fields are ", matindex.dtype)

  # The index does NOT have a consistent row or column structure, I assume 
  # the reason is just that it's composed of a bunch of different MATLAB arrays

  # The columns are transposed occasionally because otherwise they don't fit
  # together - they're imported from MATLAB in a bunch of inconsistent dimensions

  num_examples = matindex[matindex.dtype.names[1]].size

  print("There are ", num_examples, " images in the dataset")

  print("Here's a list of the attributes in the MATLAB data:")

  for name_i in matindex.dtype.names:
    print("Attribute: ", name_i)
    print("Dimensions of ", name_i ": ", matindex[name_i].shape)

  # --------

  # putting image attributes in a DataFrame

  filename_col = pd.DataFrame(matindex['filename'].T, columns=['filename'])

  folder_col = pd.DataFrame(matindex['folder'].T, columns=['folder'])

  # I don't know what this column is for (it's not documented on the dataset site)
  typeset_col = pd.DataFrame(matindex['typeset'], columns=['typeset'])

  # scene type of each image
  scene_col = pd.DataFrame(matindex['scene'].T, columns=['scene'])

  # putting the columns together
  image_index = pd.concat([filename_col, folder_col, typeset_col, scene_col], axis=1)

  # image_index.to_csv("csvIndexes/image_index.csv")

  # -------

  # Putting object attributes in a DataFrame

  object_name_list = pd.DataFrame(matindex['objectnames'].T, columns=['objectnames'])

  # object_name_list.to_csv("csvIndexes/object_names.csv")

  # ----

  # Extracting object frequency matrix (gives number of times each object in the
  # list of objects occurs in each image)
  # We could have gotten this ourselves from the text files if we wanted, but
  # the parsing format is not fun, so I decided to stick with converting the
  # MATLAB code

  # image filenames are rows, and words (object names) are columns

  object_image_matrix = pd.DataFrame(matindex['objectPresence'].T, 
                                    columns=object_name_list['objectnames'],
                                    index=filename_col['filename'])

  # Function to produce all 3 CSV files
  # THE LAST ONE IS KINDA BIG (for a CSV) - around 300 MB
  def printALLCSVs():
    image_index.to_csv("csvIndexes/image_index.csv")
    object_name_list.to_csv("csvIndexes/object_names.csv")
    object_image_matrix.to_csv("csvIndexes/object_image_matrix.csv")

def main():
  index = ADEIndex()
  if args.printALLCSVs == 'Y':
    index.printALLCSVs()	main()