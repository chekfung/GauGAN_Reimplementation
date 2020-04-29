import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub

import numpy as np

from imageio import imwrite
import os
import csv
import argparse

from code.discriminator import Discriminator 
from code.generator import SPADEGenerator
from preprocess import load_image_batch

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

## --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='GAUGAN')

parser.add_argument('--train-img-dir', type=str, default='./data/landscape_data/train',
					help='Data where training images live')

parser.add_argument('--test-img-dir', type=str, default='./data/landscape_data/test',
					help='Data where test images live')

parser.add_argument('--out-dir', type=str, default='./output',
					help='Data where sampled output images will be written')

parser.add_argument('--mode', type=str, default='train',
					help='Can be "train" or "test"')

parser.add_argument('--restore-checkpoint', action='store_true',
					help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--z-dim', type=int, default=16,
					help='Dimensionality of the latent space')

parser.add_argument('--batch-size', type=int, default=16,
					help='Sizes of image batches fed through the network')

parser.add_argument('--num-data-threads', type=int, default=10,
					help='Number of threads to use when loading & pre-processing training images')

parser.add_argument('--num-epochs', type=int, default=200,
					help='Number of passes through the training data to make before stopping')

parser.add_argument('--gen-learn-rate', type=float, default=0.0001,
					help='Learning rate for Generator Adam optimizer')

parser.add_argument('--dsc-learn-rate', type=float, default=0.0004,
					help='Learning rate for Discriminator Adam optimizer')

parser.add_argument('--beta1', type=float, default=0.5,
					help='"beta1" parameter for Adam optimizer')

parser.add_argument('--beta2', type=float, default=0.999,
					help='"beta2" parameter for Adam optimizer')
					
parser.add_argument('--img-h', type=int, default=120,
					help='height of image')
					
parser.add_argument('--img-w', type=int, default=160,
					help='width of image')

parser.add_argument('--log-every', type=int, default=7,
					help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=500,
					help='Save the state of the network after every [this many] training iterations')

parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
					help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

args = parser.parse_args()

## --------------------------------------------------------------------------------------

# Numerically stable logarithm function
def log(x):
	"""
	Finds the stable log of x
	:param x: 
	"""
	return tf.math.log(tf.maximum(x, 1e-5))

## --------------------------------------------------------------------------------------

# For evaluating the quality of generated images
# Frechet Inception Distance measures how similar the generated images are to the real ones
# https://nealjean.com/ml/frechet-inception-distance/
# Lower is better
module = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/classification/4", output_shape=[1001])])
def fid_function(real_image_batch, generated_image_batch):
	"""
	Given a batch of real images and a batch of generated images, this function pulls down a pre-trained inception 
	v3 network and then uses it to extract the activations for both the real and generated images. The distance of 
	these activations is then computed. The distance is a measure of how "realistic" the generated images are.
	:param real_image_batch: a batch of real images from the dataset, shape=[batch_size, height, width, channels]
	:param generated_image_batch: a batch of images generated by the generator network, shape=[batch_size, height, width, channels]
	:return: the inception distance between the real and generated images, scalar
	"""
	INCEPTION_IMAGE_SIZE = (299, 299)
	real_resized = tf.image.resize(real_image_batch, INCEPTION_IMAGE_SIZE)
	fake_resized = tf.image.resize(generated_image_batch, INCEPTION_IMAGE_SIZE)
	module.build([None, 299, 299, 3])
	real_features = module(real_resized)
	fake_features = module(fake_resized)
	return tfgan.eval.frechet_classifier_distance_from_activations(real_features, fake_features)

# Train the model for one epoch.
def train(generator, discriminator, dataset_iterator, manager):
	"""
	Train the model for one epoch. Save a checkpoint every 500 or so batches.
	:param generator: generator model
	:param discriminator: discriminator model
	:param dataset_ierator: iterator over dataset, see preprocess.py for more information
	:param manager: the manager that handles saving checkpoints by calling save()
	:return: The average FID score over the epoch
	"""
	# Loop over our data until we run out
	total_fid = 0
	total_gen_loss = 0
	total_disc_loss =0
	iterations = 0

	# print("dataset_iterator is ", dataset_iterator)
	# print(dataset_iterator[0])

	for iteration, batch in enumerate(dataset_iterator):

		# Break batch up into images and segmaps 
		images, seg_maps = batch

		with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

			# calculate generator output
			gen_output = generator.call(seg_maps)
			
			# Get discriminator output for fake images and real images
			disc_real = discriminator.call(images, seg_maps)
			# disc_fake = discriminator.call(gen_output, seg_maps)
			disc_fake = disc_real
			
			# calculate gen. loss and disc. loss
			g_loss = generator.loss(disc_fake)
			d_loss = discriminator.loss(disc_real, disc_fake)
			
			# Update loss counters
			total_gen_loss += g_loss
			total_disc_loss += d_loss

			# get gradients
			g_grad = generator_tape.gradient(g_loss, generator.trainable_variables)
			d_grad = discriminator_tape.gradient(d_loss, discriminator.trainable_variables)
			
		generator.optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))
		discriminator.optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))
		
		# Save
		if iteration % args.save_every == 0:
			manager.save()

		# Calculate inception distance and track the fid in order
		# to return the average
		if iteration % 500 == 0:
			fid_ = fid_function(images, gen_output)
			total_fid += fid_
			iterations += 1
			print('**** INCEPTION DISTANCE: %g ****' % fid_)
	return total_fid / iterations, total_gen_loss / iterations, total_disc_loss / iterations


# Test the model by generating some samples.
def test(generator, dataset_iterator):
	"""
	Test the model.
	:param generator: generator model
	:return: None
	"""
	total_fid = 0
	total_gen_loss = 0
	total_disc_loss = 0
	iterations = 0

	for iteration, batch in enumerate(dataset_iterator):
		image, seg_map = batch
		img = generator.call(seg_map).numpy()

		# Rescale the image from (-1, 1) to (0, 255)
		img = ((img / 2) - 0.5) * 255
		# Convert to uint8
		img = img.astype(np.uint8)
		# Save images to disk
		for i in range(0, 1):
			img_i = img[i]
			s = args.out_dir+'/'+str(i)+'_generated.png'
			s2 = args.out_dir+'/'+str(i)+'_truth.png'
			imwrite(s, img_i)
			imwrite(s2, image)

		iterations += 1
	
	return total_fid / iterations, total_gen_loss / iterations, total_disc_loss / iterations
	
## --------------------------------------------------------------------------------------

def main():
	# Load train images (to feed to the discriminator)

	train_dataset_iterator = load_image_batch(dir_name=args.train_img_dir, batch_size=args.batch_size, \
		n_threads=args.num_data_threads)
	
	# Get number of train images and make an iterator over it
	test_dataset_iterator = load_image_batch(dir_name=args.test_img_dir, batch_size=1, \
		n_threads=args.num_data_threads, drop_remainder=False)

	# Initialize generator and discriminator models
	generator = SPADEGenerator(args.beta1, args.beta2, args.gen_learn_rate, args.batch_size, args.z_dim, args.img_w, args.img_h)
	discriminator = Discriminator(args.beta1, args.beta2, args.dsc_learn_rate)

	print('========================== GENERATOR ==========================')
	# Charlie arbitrarily put 4x4 for the input dims just to see if the code would run
	# Error message: 
	''''
	  File "/gpfs/main/home/cgagnon1/course/cs1430/gaugan/James_TompGAN/code/generator.py", line 34, in call
    result_dense = self.dense(noise)

    ValueError: Input 0 of layer dense is incompatible with the layer: 
    : expected min_ndim=2, found ndim=1. Full shape received: [2]

	'''
	# generator.build(input_shape=(4,4))
	# generator.summary()
	print('========================== DISCRIMINATOR ==========================')
	# discriminator.summary()

	# For saving/loading models
	checkpoint_dir = './checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
	manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
	# Ensure the output directory exists
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	if args.restore_checkpoint or args.mode == 'test':
		# restores the latest checkpoint using from the manager
		checkpoint.restore(manager.latest_checkpoint) 

	try:
		# Specify an invalid GPU device
		with tf.device('/device:' + args.device):
			if args.mode == 'train':
				for epoch in range(0, args.num_epochs):
					print('========================== EPOCH %d  ==========================' % epoch)
					avg_fid, avg_g_loss, avg_d_loss = train(generator, discriminator, train_dataset_iterator, manager)
					print("Average FID for Epoch: " + str(avg_fid))
					# Save at the end of the epoch, too
					print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
					manager.save()
					
					# Save the losses and fid into a CSV that we make.
					logs_path = "logs"
					fn = "fid_losses_train.csv"
					full_path = logs_path + '/' + fn
					
					# Make logs directory if it does not exist
					if (not os.path.exists(logs_path)):
						os.path.makedir(logs_path)

					# If first Epoch create new file
					if epoch == 0:
						with open(full_path, 'w') as csvfile:
							csvwriter = csv.writer(csvfile)

							# Write the categories and first epoch info
							csvwriter.writerow(['Average FID', 'Average Generator Loss', 'Average Discriminator Loss'])
							csvwriter.writerow([avg_fid, avg_g_loss, avg_d_loss])
					else:
						# If any other epoch, append
						with open(full_path, 'a+', newline='') as csvfile:  
							csvwritter = csv.writer(csvfile)

							# Write epoch information
							csvwritter.writerow([avg_fid, avg_g_loss, avg_d_loss])

			if args.mode == 'test':
				avg_fid, avg_g_loss, avg_d_loss = test(generator, test_dataset_iterator)

				# Save the losses and fid into a CSV that we make.
				logs_path = "logs"
				fn = "fid_losses_test.csv"
				full_path = logs_path + '/' + fn
				
				# Make logs directory if it does not exist
				if (not os.path.exists(logs_path)):
					os.path.makedir(logs_path)

				# If first Epoch create new file
				with open(full_path, 'w') as csvfile:
					csvwriter = csv.writer(csvfile)

					# Write the categories and first epoch info
					csvwriter.writerow(['Average FID', 'Average Generator Loss', 'Average Discriminator Loss'])
					csvwriter.writerow([avg_fid, avg_g_loss, avg_d_loss])

	except RuntimeError as e:
		print(e)
	
if __name__ == '__main__':
	main()
