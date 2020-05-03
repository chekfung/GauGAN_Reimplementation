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
import tensorflow as tf

# make this accessible to entire file
adeindex = MATLABconversion.ADEIndex()

# Resize parameters
HEIGHT = 96
WIDTH = 128

# Number of object list items required for an image to be included
UNIQUE_APPROVED_OBJECTS_REQUIRED = 3

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
    else:
        orig_path = os.path.join(sys.path[0], data_set_path, 'validation')

    file_categories = []
    # filename = 'test_explicit.txt'
    filename = 'explicit_cv_landscapes_final_project.txt'

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
    
    # print(real_filepaths)
    return real_filepaths


def get_images_by_object():
    
    """
    Given objects_we_want.txt, we select images based on whether they contain relevant objects
    
    Returns: 
    :real_filepaths - .jpg filepaths to images
    :object_names - list of words of objects we want to include
    """

    object_names = []
    list_of_objects = 'objects_we_want.txt'
    # list_of_objects = 'test_object_selection.txt'

    # Get all the object names that we want
    with open(os.path.join(sys.path[0], list_of_objects)) as f:
        for line in f:
            if line != '\n':
                object_names.append(line.strip())

    # Now, for each of these object names, go in and grab all image filepaths that contain that object
    real_filepaths = set()

    for name in object_names:

        #print("Getting files that contain: " + name)
        
        # adeindex = MATLABconversion.ADEIndex()

        object_image_matrix = adeindex.object_image_matrix
        image_stats_matrix = adeindex.image_index

        # get all columns where the column title contains name (the object name we're looking for) in a piece of the tile separated by ", " 
        object_cols_that_match = object_image_matrix.loc[:,[string for string in object_image_matrix.columns if name in string.split(", ")]]

        # for loop needed here because term could match more than one object category
        for (colName, colData) in object_cols_that_match.iteritems():
            image_rows_to_add = object_image_matrix.loc[object_image_matrix[colName] != 0]
            
            # print("img rows to add are ", image_rows_to_add)

            for index, row in image_rows_to_add.iterrows():
                # print('looking at index of matched images that is #:', index)
                filepath = None
                if adeindex.CSVsExist:
                    filepath = image_stats_matrix.loc[index,'folder'] + '/' + image_stats_matrix.loc[index,'filename']
                else:
                    filepath = image_stats_matrix.loc[index,'folder'] + '/' + index
                # print(filepath)
                real_filepaths.add(filepath)

    #print(real_filepaths)
    return real_filepaths, object_names


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
    #print("Loading from ", img_filepath)
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
        #print(filepath)
        segpath = filepath[:-4] + '_seg.png'

        if 'validation' in filepath:
            img_test.append(filepath)
            seg_test.append(segpath)
        else:
            img_train.append(filepath)
            seg_train.append(segpath)

    return img_train, seg_train, img_test, seg_test

def save_shrunken_image(img, train_dir, test_dir, whether_training):

    # Skip over any files that have parts_1 or parts_2 in filename
    if '_parts_' in img:
        return

    filename = os.path.basename(img)

    # load each image and put in correct folder
    resized = load_img(img, HEIGHT, WIDTH)

    if whether_training:
        f = os.path.join(train_dir, filename)
        imsave(f, img_as_ubyte(resized))
        #print("Saving training image " + f)
    else:
        f = os.path.join(test_dir, filename)
        imsave(f, img_as_ubyte(resized))
        #print("Saving testing image " + f)

def save_shrunken_segmap(img, approved_words, train_dir, test_dir, whether_training):
    
    # Skip over any files that have parts_1 or parts_2 in filename
    if '_parts_' in img:
        return

    # DONE: Depending on how testing works, implement the seg map values from 
    # 0 to n where 0 represents bad values that we do not want (achieved with
    # 255 multiplication and int casting and 255 mult. in innermost for loop)
    filename = os.path.basename(img)
    #print(filename)
    total_num_approved_words = len(approved_words)
    # load each segmap in full size to knock out irrelevant objects

    initial_segmap = imread(img)

    r = initial_segmap[:,:,0]
    g = initial_segmap[:,:,1]

    r = r.astype(np.uint16)
    g = g.astype(np.uint16)

    object_map =  np.floor((r / 10)) * 256 + g
    whether_values_are_zero = object_map == 0
    # to fix MATLAB indexing
    object_map[whether_values_are_zero] = 1

    # get unique image labels in the decoded segmap (list of contained objects)
    unique_obj_codes = np.unique(object_map)
    # if object is not in our list of approved objects, then set all pixels of this object to 0
    num_approved_words_in_img = 0  
    for code in unique_obj_codes:
        # the MATLAB indexing fix described above /should/ prevent this from being a key error
        img_object_name = adeindex.object_name_list['objectnames'].loc[code - 1]

        approved_code = False
        # Checking if current object is on our list of approved words
        for word_index, word in enumerate(approved_words):
            # This "in" is checking list containment (img_object_name.split(", ")) is a list of strings
            if word in img_object_name.split(", "):
                parts_of_object_map_with_this_object = object_map == code
                # TODO: make the conversion to 0-255 safer (no more int cast rounding)
                object_map[parts_of_object_map_with_this_object] = int(255*(word_index + 1)/total_num_approved_words)
                approved_code = True
                break
        
        if approved_code:
            # print(img_object_name + " is approved")
            num_approved_words_in_img = num_approved_words_in_img + 1
            # do not change pixel values for approved objects
        else:
            parts_of_object_map_with_this_object = object_map == code
            object_map[parts_of_object_map_with_this_object] = 0
            
    # Image does not contain sufficient number of different objects 
    # -> don't include it
    if num_approved_words_in_img < UNIQUE_APPROVED_OBJECTS_REQUIRED:
        # print("Tossing out " + filename + " because it doesn't have enough unique objects on our list")
        if whether_training:
            os.remove(os.path.join(train_dir, filename[:-8] + '.jpg'))
        else:
            os.remove(os.path.join(test_dir, filename[:-8] + '.jpg'))
        return

        '''
         This^^ implementation leaves object codes in their original integer encodings 
         --> DOES NOT re-enumerate from 0 to n objects, because counting how many objects 
            our words match to, and then enumerating the matches, is a pain
        '''

    # Now, object_map has nonzero pixel values only for objects that we care about
    object_map = object_map.astype(np.uint8)

    resized_segmap = tf.image.resize(object_map[:,:,np.newaxis], size=(HEIGHT, WIDTH), method='nearest')[:,:,0]
    npy_segmap = np.array(resized_segmap)
    generic_filename, ext = os.path.splitext(filename)
    
    if whether_training:
        f = os.path.join(train_dir, filename)
        npy_path = os.path.join(train_dir,generic_filename)
        # print("Resized segmap is ", resized_segmap)
        # print("Shape is ", resized_segmap.shape)
        imsave(f, resized_segmap)
        # np.savetxt(npy_path + '.csv', npy_segmap, delimiter=",")
        #print("Saving training segmap " + f)
    else:
        f = os.path.join(test_dir, filename)
        npy_path = os.path.join(test_dir,generic_filename)
        imsave(f, resized_segmap)
        # np.savetxt(npy_path + '.csv', npy_segmap, delimiter=",")
        #print("Saving testing segmap " + f)

def main():        
    # Create the file directories to house the new resized imgs
    file_dir = 'landscape_data'

    train_dir, test_dir = make_save_dir(file_dir)
    
    data_set_path = os.path.join('ADE20K_2016_07_26', 'images')
    
    # Get list of objects we want and filepaths for object-wise selection
    files_by_object, object_names = get_images_by_object()

    # Add Training images by explicit scene - from ADE20K Train set
    explicit_filepaths = find_explicit_files(data_set_path, train=True)
    
    for filepath in explicit_filepaths:
        imgs, segs = get_explicit_files(filepath)

        for img in imgs:
            save_shrunken_image(img, train_dir, test_dir, whether_training=True)
        
        for seg in segs:
            # Sets all segmap regions that contain objects NOT in our list of
            # relevant objects to pixel value 0
            save_shrunken_segmap(seg, object_names, train_dir, test_dir, whether_training=True)

    print("Done loading resized Training data selected explicitly by scene")

    # Add Testing images by explicit scene - from ADE20K Validation set
    explicit_filepaths = find_explicit_files(os.path.join('ADE20K_2016_07_26', 'images'), train=False)

    # For the testing/validation set
    for filepath in explicit_filepaths:
        imgs, segs = get_explicit_files(filepath)

        for img in imgs:
            save_shrunken_image(img, train_dir, test_dir, whether_training=False)
        for seg in segs:
            save_shrunken_segmap(seg, object_names, train_dir, test_dir, whether_training=False)

    # Remove parts2 files (necessary for explicit scene selection)

    remove_parts_two(test_dir)
    remove_parts_two(train_dir)

    print("Done loading resized Testing data selected explicitly by scene")

    training_images, training_segs, testing_images, testing_segs = split_files_by_object(files_by_object)

    for img in training_images:
        save_shrunken_image(img, train_dir, test_dir, whether_training=True)
    
    for seg in training_segs:
        save_shrunken_segmap(seg, object_names, train_dir, test_dir, whether_training=True)

    print("Done loading resized Training data selected by object content")

    for img in testing_images:
        save_shrunken_image(img, train_dir, test_dir, whether_training=False) 

    for seg in testing_segs:
        save_shrunken_segmap(seg, object_names, train_dir, test_dir, whether_training=False)
    
    print("Done loading resized Testing data selected by object content")

if __name__ == "__main__":
    main()





