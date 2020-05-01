from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
import pandas as pd
import sys
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='GAUGAN')

parser.add_argument('--saveCSVs', type=str, default='N',
					help='Whether to print CSV files from MATLAB index (Y or N)')

csv_folderpath = os.path.join(sys.path[0], 'csvIndexes')

args = parser.parse_args()

class ADEIndex():

  def __init__(self):
    self.image_index = None
    self.object_name_list = None
    self.object_image_matrix = None
    self.CSVsExist = False

    if os.path.exists(csv_folderpath):

      print("Now loading data from CSV files")
      self.image_index = pd.read_csv(os.path.join(csv_folderpath, 'image_index.csv'))
      self.object_name_list = pd.read_csv(os.path.join(csv_folderpath, 'object_name_list.csv'))
      self.object_image_matrix = pd.read_csv(os.path.join(csv_folderpath, 'object_image_matrix.csv'))
      self.CSVsExist = True

    else:

      print("No CSVs found - will save CSVs after loading MATLAB data")

      # This script should be run from within the project's James_TompGAN/data/ folder

      # data_dir = pjoin('ADE20K_2016_07_26')

      # mat_fname = pjoin(data_dir, 'index_ade20k.mat')

      mat_fname = os.path.join(sys.path[0], 'ADE20K_2016_07_26', 'index_ade20k.mat')

      mat_contents = sio.loadmat(mat_fname)

      matindex = mat_contents['index'][0,0]

      # print("Index fields are ", matindex.dtype)

      # The index does NOT have a consistent row or column structure, I assume 
      # the reason is just that it's composed of a bunch of different MATLAB arrays

      # The columns are transposed occasionally because otherwise they don't fit
      # together - they're imported from MATLAB in a bunch of inconsistent dimensions

      num_examples = matindex[matindex.dtype.names[1]].size

      print("There are ", num_examples, " images in the dataset")

      # print("Here's a list of the attributes in the MATLAB data:")

      # for name_i in matindex.dtype.names:
      #   print("Attribute: ", name_i)
      #   print("Dimensions of ", name_i, ": ", matindex[name_i].shape)

      # --------

      # putting image attributes in a DataFrame

      filename_col_nested = pd.DataFrame(matindex['filename'].T, columns=['filename'])
      
      filename_col = pd.DataFrame(columns=['filename'])

      for index, row in filename_col_nested.iterrows():
        filename_col.loc[index] = filename_col_nested['filename'][index][0]

      folder_col_nested = pd.DataFrame(matindex['folder'].T, columns=['folder'])

      folder_col = pd.DataFrame(columns=['folder'])

      for index, row in folder_col_nested.iterrows():
        folder_col.loc[index] = folder_col_nested['folder'][index][0]

      # I don't know what this column is for (it's not documented on the dataset site)
      typeset_col = pd.DataFrame(matindex['typeset'], columns=['typeset'])

      # scene type of each image
      scene_col = pd.DataFrame(matindex['scene'].T, columns=['scene'])

      # putting the columns together
      int_indexed_image_index = pd.concat([filename_col, folder_col, typeset_col, scene_col], axis=1)

      self.image_index = int_indexed_image_index.set_index('filename')
      # print(image_index.index)
      # print(image_index)
      # print(image_index['ADE_train_00011093.jpg'])

      # image_index.to_csv("csvIndexes/image_index.csv")

      # print(image_index['ADE_train_00011093.jpg'])

      # -------

      # Putting object attributes in a DataFrame

      object_name_list_nested = pd.DataFrame(matindex['objectnames'].T, columns=['objectnames'])

      self.object_name_list = pd.DataFrame(columns=['objectnames'])

      for index, row in object_name_list_nested.iterrows():
        self.object_name_list.loc[index] = object_name_list_nested['objectnames'][index][0]

      # object_name_list.to_csv("csvIndexes/object_names.csv")

      # ----

      # Extracting object frequency matrix (gives number of times each object in the
      # list of objects occurs in each image)
      # We could have gotten this ourselves from the text files if we wanted, but
      # the parsing format is not fun, so I decided to stick with converting the
      # MATLAB code

      # image filenames are rows, and words (object names) are columns

      self.object_image_matrix = pd.DataFrame(matindex['objectPresence'].T, 
                                        columns=self.object_name_list['objectnames'],
                                        index=filename_col['filename'])

      # object_cols_that_match = object_image_matrix.loc[:,[x for x in object_image_matrix.columns if 'vcr' in x]]
      # for (colName, colData) in object_cols_that_match.iteritems():
      #   image_rows_to_add = object_image_matrix.loc[object_image_matrix[colName] != 0]
      #   print(image_rows_to_add)

  # Function to produce all 3 CSV files
  # THE LAST ONE IS KINDA BIG (for a CSV) - around 300 MB
  def saveALLCSVs(self):
    self.image_index.to_csv(os.path.join("csvIndexes","image_index.csv"))
    self.object_name_list.to_csv(os.path.join('csvIndexes', 'object_name_list.csv'))
    self.object_image_matrix.to_csv(os.path.join("csvIndexes","object_image_matrix.csv"))

def main():
  index = ADEIndex()
  if args.saveCSVs == 'Y' or index.CSVsExist == False:
    if os.path.exists(csv_folderpath):
      shutil.rmtree(csv_folderpath)
    os.mkdir(csv_folderpath)
    print("Now printing CSV files")
    index.saveALLCSVs()
    print("Your CSV files are now toasty and warm")

if __name__ == '__main__':
  main()
