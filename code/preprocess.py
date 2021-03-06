import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
import sys

import matplotlib.pyplot as plt

"""
MODELED AFTER Brown CSCI 1470 DEEP LEARNING GAN ASSIGNMENT 7 HOMEWORK
"""

# Sets up tensorflow graph to load images
# (This is the version using new-style tf.data API)
def load_image_batch(dir_name, batch_size=32, shuffle_buffer_size=25, n_threads=10, drop_remainder=True):
    """
    Given a directory and a batch size, the following method returns a dataset iterator that can be queried for 
    a batch of images

    :param dir_name: a batch of images
    :param batch_size: the batch size of images that will be trained on each time
    :param shuffle_buffer_size: representing the number of elements from this dataset from which the new dataset will 
    NOTE: At the moment, we do not know if we need to change this ^
    sample
    :param n_thread: the number of threads that will be used to fetch the data

    :return: an iterator into the dataset
    """
    objects_file = './data/objects_we_want.txt'
    f = open(objects_file, 'r')

    # Add plus one due to the zero that represents the all other objects
    num_objects = len(f.read().split('\n')) + 1

    # Function used to load and pre-process image files
    # (Have to define this ahead of time b/c Python does allow multi-line
    #    lambdas, *grumble*)
    def load_and_process_image(file_path):
        """
        Given a file path, this function opens and decodes the image stored in the file.

        :param file_path: a batch of images

        :return: an rgb image
        """
        # Load image
        image = tf.io.decode_png(tf.io.read_file(file_path), channels=3)

        # Convert image to normalized float (0, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Rescale data to range (-1, 1)
        #image = (image - 0.5) * 2
        return image
    
    def load_and_process_segmap(file_path):
        """
        Given a file path, this function opens and decodes the image stored in the file.

        :param file_path: a batch of images

        :return: a one-hot encoded segmap
        """
        # Load image
        # Grayscale already, so transform to 2D grayscale array
        image = tf.io.decode_png(tf.io.read_file(file_path), channels=1) 
        image = tf.squeeze(image)
        # Charlie does not think we should be normalizing the segmaps
        # # Convert image to normalized float (0, 1)
        image = tf.image.convert_image_dtype(image, tf.uint8) * num_objects
        image = tf.cast(image, tf.int32)
        original_shape = tf.shape(image)
        image = tf.reshape(image, shape=(-1,))
        _, idx = tf.unique(image)
        image = tf.reshape(idx, original_shape)
        one_hot = tf.one_hot(image, num_objects)

        # Rescale data to range (-1, 1)
        #image = (image - 0.5) * 2
        return one_hot
    
    def augment(image, segmap):
        """
        Given an image-segmap pair, apply identical image augmentation

        :param image: decoded tf images

        :param segmap: decoded segmap tf images
 
        :return: augmented (image, segmap) tuple, both decoded
        """
        # Flip image horizontally
        flip_bool = np.random.rand()

        if (flip_bool >= 0.5):
            image = tf.image.flip_left_right(image)
            segmap = tf.image.flip_left_right(image)

        # Rotate image (15-20 degrees or so)
        deg_range = 15 
        rotation_angle = np.random.randint(-np.radians(deg_range),np.radians(deg_range))
        image = tfa.image.rotate(image, rotation_angle)
        segmap = tfa.image.rotate(segmap, rotation_angle)

        # Randomly crop images
        stacked_image = tf.stack([image, segmap], axis=0)

        # Note: Technically should be 160 by 120, so if we need to hardcode it, we can hardcode it.
        cropped_stack = tf.image.random_crop(stacked_image, size=[2, image.shape[0], image.shape[1], 3])

        aug_image = cropped_stack[0]
        aug_segmap = cropped_stack[1]

        return (aug_image, aug_segmap)

    def get_image_segmap_pair(segmap_path):
        """
        Given a filepath for a segmap, this function gets the corresponding segmap and returns the (image, segmap) tuple after decoding both.

        :param segmap_path: a filepath to one segmap 

        :return: (image, segmap) tuple, both decoded
        """

        segmap_path_len = tf.strings.length(segmap_path)

        # args are string tensor, start index, and length
        image_name_without_ext = tf.strings.substr(segmap_path, 0, segmap_path_len - 8)

        image_path = tf.strings.join([image_name_without_ext, '.jpg'])

        # image_path = segmap_path[:-8] + '.png'
        # image_path = 'a'

        # Load in image pair to return

        # # The approach below doesn't work anymore because when images are saved
        # # as segmaps, their pixel value ranges get clipped
        # segmap = load_and_process_image(segmap_path)

        
        ## This approach fails because np.load() cannot be called on a Tensor string
        # with tf.compat.v1.Session() as sess:
        #     segmap = tf.convert_to_tensor(np.load(sess.run(segmap_path)))
        # image = load_and_process_image(image_path)


        segmap = load_and_process_segmap(segmap_path)

        image = load_and_process_image(image_path)


        ## For data augmentation:
        # augmented_pair = augment(image, segmap)

        # return augmented_pair
        return image, segmap

    
    # RegEx to match all segmap paths
   
    ## Approach 1: save segmaps as shrunken png files
    ## Problem: 
    # seg_path = dir_name + '/*_seg.png'

    ## Approach 2: save segmaps as .npy files
    ## Problem: can't call np.load() on a Tensor string
    # seg_path = dir_name + '/*.npy'

    ## Approach 3: save segmaps as csv files and load with tf
    ## Fails because tf loads csv files by separate Tensors by column (not a unified array)
    # seg_path = dir_name + '/*.csv'


    # Current approach: save segmaps as png images after reassigning all object
    # encodings 
    seg_path = dir_name + '/*.png'

    dataset = tf.data.Dataset.list_files(seg_path)

    # Shuffle order
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Load and process images (in parallel)
    dataset = dataset.map(map_func=get_image_segmap_pair, num_parallel_calls=n_threads)

    # Create batch, dropping the final one which has less than batch_size elements and finally set to reshuffle
    # the dataset at the end of each iteration
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Prefetch the next batch while the GPU is training
    dataset = dataset.prefetch(1)

    # Return an iterator over this dataset
    return dataset
