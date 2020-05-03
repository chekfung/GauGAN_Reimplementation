import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Layer
from code.spectral_norm import spectral_conv

class SpadeLayer(Layer):
	def __init__(self, in_channels, out_channels, use_bias=True, hidden_channels=128):
		super(SpadeLayer, self).__init__()
		self.bn = BatchNormalization()
		self.glorot = tf.keras.initializers.GlorotNormal()
		# Kernel=5, Strides=1, out_channels=hidden_channels
		self.conv0 = tf.Variable(self.glorot(shape=[5,5,in_channels, hidden_channels])) 
		self.bias0 = tf.Variable(self.glorot(shape=[hidden_channels]))
		self.relu = ReLU()
		# Kernel=5, strides=1, out_channels=out_channels
		self.conv1 = tf.Variable(self.glorot(shape=[5,5,hidden_channels, out_channels])) 
		self.bias1 = tf.Variable(self.glorot(shape=[out_channels]))
		# kernel=5, strides=1, out_channels=out_channels
		self.conv2 = tf.Variable(self.glorot(shape=[5,5,hidden_channels, out_channels])) 
		self.bias2 = tf.Variable(self.glorot(shape=[out_channels]))


	""" def build(self, input_shape): 
		super(SpadeLayer, self).build(input_shape) """

	def call(self, features, segmap):
		norm = self.bn(features)
		
		_, x_h, x_w, _ = list(norm.shape)
		segmap_resized = tf.image.resize(segmap, size=(x_h, x_w), method="nearest")

		seg_result = spectral_conv(inputs=segmap_resized, weight=self.conv0, stride=1, bias=self.bias0)
		seg_result = self.relu(seg_result)
		result_a = spectral_conv(inputs=seg_result, weight=self.conv1, stride=1, bias=self.bias1)
		result_b = spectral_conv(inputs=seg_result, weight=self.conv2, stride=1, bias=self.bias2)

		x = tf.math.add(1.0, result_a)
		x = tf.multiply(x, norm)
		x = tf.math.add(x, result_b)
		return x