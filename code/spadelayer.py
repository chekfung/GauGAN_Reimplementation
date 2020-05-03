import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Layer
from code.spectral import spectral_norm

class SpadeLayer(Layer):
	def __init__(self, out_channels, use_bias=True, hidden_channels=128):
		super(SpadeLayer, self).__init__()
		self.bn = BatchNormalization()
		self.conv1 = Conv2D(filters=hidden_channels, kernel_size=5, strides=1, padding="SAME", \
			use_bias=use_bias, dtype=tf.float32, kernel_initializer=tf.random_normal_initializer(stddev=.02))
		self.relu = ReLU()
		self.conv2 = Conv2D(filters=out_channels, kernel_size=5, strides=1, padding="SAME", \
			use_bias=use_bias, dtype=tf.float32, kernel_initializer=tf.random_normal_initializer(stddev=.02)) 
		self.conv3 = Conv2D(filters=out_channels, kernel_size=5, strides=1, padding="SAME", \
			use_bias=use_bias, dtype=tf.float32, kernel_initializer=tf.random_normal_initializer(stddev=.02))


	""" def build(self, input_shape): 
		super(SpadeLayer, self).build(input_shape) """

	def call(self, features, segmap):
		norm = self.bn(features)
		
		_, x_h, x_w, _ = list(norm.shape)
		segmap_resized = tf.image.resize(segmap, size=(x_h, x_w), method="nearest")

		seg_result = spectral_norm(self.conv1(segmap_resized))
		seg_result = self.relu(seg_result)
		result_a = spectral_norm(self.conv2(seg_result))
		result_b = spectral_norm(self.conv3(seg_result))

		x = tf.math.add(1.0, result_a)
		x = tf.multiply(x, norm)
		x = tf.math.add(x, result_b)
		return x