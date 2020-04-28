import numpy as np
import tensorflow as tf
import os

"""
MODELED AFTER DEEP LEARNING GAN ASSIGNMENT 7 HOMEWORK
"""

# Sets up tensorflow graph to load images
# (This is the version using new-style tf.data API)
def load_image_batch(dir_name, batch_size=32, shuffle_buffer_size=250000, n_threads=2):
    """
    Given a directory and a batch size, the following method returns a dataset iterator that can be queried for 
    a batch of images

    :param dir_name: a batch of images
    :param batch_size: the batch size of images that will be trained on each time
    :param shuffle_buffer_size: representing the number of elements from this dataset from which the new dataset will 
    sample
    :param n_thread: the number of threads that will be used to fetch the data

    :return: an iterator into the dataset
    """
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
        image = (image - 0.5) * 2
        return image

    def get_image_segmap_pair(segmap_path):
        """
        Given a filepath for a segmap, this function gets the corresponding segmap and returns the (image, segmap) tuple after decoding both.

        :param file_path: a batch of images

        :return: (image, segmap) tuple, both decoded
        """
        image_path = segmap_path[:-8] + '.png'

        # Load in image pair to return
        segmap = load_and_process_image(segmap_path)
        image = load_and_process_image(image_path)

        return (image, segmap)

    
    # RegEx to match all segmap paths
    seg_path = dir_name + '/*_seg.png'
    dataset = tf.data.Dataset.list_files(seg_path)

    # Shuffle order
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Load and process images (in parallel)
    dataset = dataset.map(map_func=get_image_segmap_pair, num_parallel_calls=n_threads)

    # Create batch, dropping the final one which has less than batch_size elements and finally set to reshuffle
    # the dataset at the end of each iteration
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch the next batch while the GPU is training
    dataset = dataset.prefetch(1)

    # Return an iterator over this dataset
    return dataset
