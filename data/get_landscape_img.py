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
import convertMATIndexToCSV as matindex

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
    filename = 'cv_landscapes_final_project.txt'

    # Get all the file categories that we want (Should be 47)
    with open(os.path.join(sys.path[0], filename)) as f:
        for line in f:
            if line != '\n':
                file_categories.append(line.strip())

    # Now, for each of these filecategories, go in and grab their actual destinations
    real_filepaths = []

    for file in file_categories:
        # outliers is an exception, so check if outliers first
        if file[0:8] == 'outliers':
            path = file
        else:
            path = os.path.join(str(file[0]), file)
        
        real_filepaths.append(os.path.join(orig_path, path))
    
    return real_filepaths


def get_images_by_object(data_set_path, train=True):
    
    # Given objects_we_want.txt, we select images based on whether they contain relevant objects

    if train:
        orig_path = os.path.join(sys.path[0], data_set_path, 'training')
        print(orig_path)
    else:
        orig_path = os.path.join(sys.path[0], data_set_path, 'validation')

    object_names = []
    filename = 'objects_we_want.txt'

    # Get all the object names that we want
    with open(os.path.join(sys.path[0], filename)) as f:
        for line in f:
            if line != '\n':
                object_names.append(line.strip())

    # Now, for each of these object names, go in and grab all image filepaths that contain that object
    real_filepaths = set()

    for obj in object_names:
        
        # if column of this object has a nonzero value, corresponding image should be added
        obj_col = matindex.object_image_matrix[obj]
        whether_image_has_object = not (obj_col == 0)
        containing_images = obj_col[whether_image_has_object].index.flatten()
        full_image_paths = 
        real_filepaths.add()
        
    real_filepaths.append(os.path.join(orig_path, path))
    
    return real_filepaths


def get_files(file_path):
    '''
    Provided a directory filepath, grabs all of the segmentation maps of the 
    images and all of the actual imgs paths.

    Params
    - file_path, one filepath representing folder that contains imgs.
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

def load_segmap(seg_filepath, h, w):
    '''
    Loads img and then resizes to the specified h, w before returning from the
    function
    '''
    seg = imread(seg_filepath)
    return resize(seg, (h, w), anti_aliasing=True)

def load_img(img_filepath, h, w):
    '''
    Loads img and then resizes to the specified h,w before returning from the 
    function
    '''
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

def main():        
    # Resize parameters
    HEIGHT = 120
    WIDTH = 160
    # Create the file directories to house the new resized imgs
    file_dir = 'landscape_data'
    train, test = make_save_dir(file_dir)
    print(train)
    print(test)

    data_set_path = os.path.join('ADE20K_2016_07_26', 'images')

    # Get the filepaths of the imgs that we want for train
    filepaths = find_explicit_files(data_set_path)

    # filepaths.append(get_images_by_object(data_set_path))
    
    for filepath in filepaths:
        imgs, segs = get_files(filepath)

        for img in imgs:
            filename = os.path.basename(img)

            # load each image and put in correct folder
            resized = load_img(img, HEIGHT, WIDTH)
            f = train + '/' + filename
            imsave(f, img_as_ubyte(resized))
        
        for seg in segs:
            filename = os.path.basename(seg)

            # Load seg map
            resized = load_segmap(seg, HEIGHT, WIDTH)
            f = train + '/' + filename
            imsave(f, img_as_ubyte(resized))
        
    remove_parts_two(train)


    print("Done loading resized Training data")

    filepaths = find_explicit_files(os.path.join('ADE20K_2016_07_26', 'images'), train=False)
    # For the testing/validation set
    for filepath in filepaths:
        imgs, segs = get_files(filepath)

        for img in imgs:
            filename = os.path.basename(img)

            # load each image and put in correct folder
            resized = load_img(img, HEIGHT, WIDTH)
            f = test + '/' + filename
            imsave(f, img_as_ubyte(resized))
        
        for seg in segs:
            filename = os.path.basename(seg)

            # Load seg map
            resized = load_segmap(seg, HEIGHT, WIDTH)
            f = test + '/' + filename
            imsave(f, img_as_ubyte(resized))
    
    remove_parts_two(test)

    print("Done loading resized Testing Data")
    

if __name__ == "__main__":
    main()





