import os
import sys
import json
import glob
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.transform import resize
import scipy.io as sio
import pandas as pd
import convertMATIndexToCSV as MATLABconversion

# Resize parameters
HEIGHT = 30
WIDTH = 40

# Schema to separate the files from each other.

# They will all be in either train or validation sets.
# Then, will be labeled with unique ID where stored in json if we need to access the files
# Segmaps will be in one folder with same ID and name; Images will be in another folder

def find_explicit_files(data_set_path, train=True):
    # NOTE: default data_set_path is: ADE20K_2016_07_26/images
    '''
    Given the cv_landscapes_final_project.txt, which contains all of the possible
    scene categories that we may want to consider, we select images by scene category
    '''
    # Set this as the path to your ADE20k Dataset
    if train:
        orig_path = os.path.join(sys.path[0], data_set_path, 'training')
        print(orig_path)
    else:
        orig_path = os.path.join(sys.path[0], data_set_path, 'validation')

    file_categories = []
    # filename = 'explicit_cv_landscapes_final_project.txt'
    filename = 'test_explicit.txt'

    # Get all the file categories that we want (Should be 47)
    with open(os.path.join(sys.path[0], filename)) as f:
        for line in f:
            if line != '\n':
                file_categories.append(line.strip())

    # Now, for each of these filecategories, go in and grab their actual destinations
    real_filepaths = set()

    for file in file_categories:
        # outliers is an exception, so check if outliers first
        if file[0:8] == 'outliers':
            path = file
        else:
            path = os.path.join(str(file[0]), file)
        
        real_filepaths.add(os.path.join(orig_path, path))
    
    print(real_filepaths)
    return real_filepaths


def get_images_by_object():
    
    # Given objects_we_want.txt, we select images based on whether they contain relevant objects

    object_names = []
    # list_of_objects = 'objects_we_want.txt'
    list_of_objects = 'test_object_selection.txt'

    # Get all the object names that we want
    with open(os.path.join(sys.path[0], list_of_objects)) as f:
        for line in f:
            if line != '\n':
                object_names.append(line.strip())

    # Now, for each of these object names, go in and grab all image filepaths that contain that object
    real_filepaths = set()

    for name in object_names:

        print("Getting files that contain: " + name)
        
        adeindex = MATLABconversion.ADEIndex()

        object_image_matrix = adeindex.object_image_matrix
        image_stats_matrix = adeindex.image_index

        object_cols_that_match = object_image_matrix.loc[:,[x for x in object_image_matrix.columns if name in x]]

        # for loop needed here because term could match more than one object category
        for (colName, colData) in object_cols_that_match.iteritems():
            image_rows_to_add = object_image_matrix.loc[object_image_matrix[colName] != 0]
            
            # print("img rows to add are ", image_rows_to_add)

            for index, row in image_rows_to_add.iterrows():
                # print('looking at index of matched images that is #:', index)
                filepath = image_stats_matrix.loc[index,'folder'] + '/' + index
                # print(filepath)
                real_filepaths.add(filepath)

    print(real_filepaths)
    return real_filepaths


def get_explicit_files(file_path):
    '''
    Provided a directory filepath, grabs all of the segmentation maps of the 
    images and all of the actual imgs paths.

    Params
    - file_path, one filepath representing folder that contains imgs.
    

    Returns

    Complete filepaths to images and segmaps
    '''
    seg_path = os.path.join(file_path, '*.png')
    img_path = os.path.join(file_path, '*.jpg')

    # Get a list of filepaths representing everything with those labels
    segs = glob.glob(seg_path)
    imgs = glob.glob(img_path)

    # Get rid of 'parts' imgs
    parts = re.compile('part')
    for seg in segs:
        if parts.search(seg):
            segs.remove(seg)

    for img in imgs:
        if parts.search(img):
            imgs.remove(img)
        
    return imgs, segs

def load_img(img_filepath, h, w):
    '''
    Loads img and then resizes to the specified h,w before returning from the 
    function
    '''
    print("Loading from ", img_filepath)
    img = imread(img_filepath)
    return resize(img ,(h,w), anti_aliasing=True)

def delete_past_dir(data_dir):
    try:
        shutil.rmtree(data_dir)
    except OSError as e:
        print("Error: %s : %s" % (data_dir, e.strerror))

def make_save_dir(file_dir):
    # Delete anything in past directory first
    delete_past_dir(file_dir)

    # Get file directory names
    test = file_dir + '/test'
    train = file_dir + '/train'

    # Make the directories
    os.makedirs(train)
    os.makedirs(test)

    # Return the names of the directories made
    return train, test

def remove_parts_two(dir):
    dir = dir + '/*_parts_2.png'
    lst = glob.glob(dir)

    for item in lst:
        os.remove(item)

# Split files-by-object into training and test sets
def split_files_by_object(files_by_object):
    img_train = []
    seg_train = []
    img_test = []
    seg_test = []

    for filepath in files_by_object:
        print(filepath)
        segpath = filepath[:-4] + '_seg.png'

        if 'validation' in filepath:
            img_test.append(filepath)
            seg_test.append(segpath)
        else:
            img_train.append(filepath)
            seg_train.append(segpath)

    return img_train, seg_train, img_test, seg_test

def save_shrunken_image(img, file_dir, train_dir, test_dir, whether_training):

    filename = os.path.basename(img)

    # load each image and put in correct folder
    resized = load_img(img, HEIGHT, WIDTH)

    if whether_training:
        f = train_dir + '/' + filename
        imsave(f, img_as_ubyte(resized))
        print("Saving training image " + f)
    else:
        f = test_dir + '/' + filename
        imsave(f, img_as_ubyte(resized))
        print("Saving testing image " + f)

def main():        
    # Create the file directories to house the new resized imgs
    file_dir = 'landscape_data'

    train_dir, test_dir = make_save_dir(file_dir)
    
    data_set_path = os.path.join('ADE20K_2016_07_26', 'images')

    # Add Training images by explicit scene - from ADE20K Train set
    explicit_filepaths = find_explicit_files(data_set_path, train=True)
    
    for filepath in explicit_filepaths:
        imgs, segs = get_explicit_files(filepath)

        for img in imgs:
            save_shrunken_image(img, file_dir, train_dir, test_dir, whether_training=True)
        
        for seg in segs:
            save_shrunken_image(seg, file_dir, train_dir, test_dir, whether_training=True)

    print("Done loading resized Training data selected explicitly by scene")

    # Add Testing images by explicit scene - from ADE20K Validation set
    explicit_filepaths = find_explicit_files(os.path.join('ADE20K_2016_07_26', 'images'), train=False)

    # For the testing/validation set
    for filepath in explicit_filepaths:
        imgs, segs = get_explicit_files(filepath)

        for img in imgs:
            save_shrunken_image(img, file_dir, train_dir, test_dir, whether_training=False)
        for seg in segs:
            save_shrunken_image(seg, file_dir, train_dir, test_dir, whether_training=False)

    # Remove parts2 files (necessary for explicit scene selection)

    remove_parts_two(test_dir)
    remove_parts_two(train_dir)

    print("Done loading resized Testing data selected explicitly by scene")

    # Add images by object content
    # List of .jpg images that contain content we want
    files_by_object = get_images_by_object()

    training_images, training_segs, testing_images, testing_segs = split_files_by_object(files_by_object)

    for img in training_images:
        save_shrunken_image(img, file_dir, train_dir, test_dir, whether_training=True)
    
    for seg in training_segs:
        save_shrunken_image(seg, file_dir, train_dir, test_dir, whether_training=True)

    print("Done loading resized Training data selected by object content")

    for img in testing_images:
        save_shrunken_image(img, file_dir, train_dir, test_dir, whether_training=False) 

    for seg in testing_segs:
        save_shrunken_image(seg, file_dir, train_dir, test_dir, whether_training=False)
    
    print("Done loading resized Testing data selected by object content")

if __name__ == "__main__":
    main()





