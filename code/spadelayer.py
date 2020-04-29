import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Layer

class SpadeLayer(Layer):
	def __init__(self, out_channels, use_bias=True, hidden_channels=128):
		super(SpadeLayer, self).__init__()
		""" self.bn = BatchNormalization()
		self.conv1 = Conv2D(filters=hidden_channels, kernel_size=5, strides=1, padding="SAME", use_bias=use_bias)
		self.relu = ReLU()
		self.conv2 = Conv2D(filters=out_channels, kernel_size=5, strides=1, padding="SAME", use_bias=use_bias) """
		self.conv3 = Conv2D(filters=out_channels, kernel_size=5, strides=1, padding="SAME", use_bias=use_bias)

	def build(self, input_shape): 
		super(SpadeLayer, self).build(input_shape)

	def call(self, features, segmap):
		""" norm = self.bn(features)
		
		_, x_h, x_w, _ = list(norm.shape)
		segmap_resized = tf.image.resize(segmap, size=(x_h, x_w), method="nearest")

		seg_result = self.relu(self.conv1(segmap_resized))
		result_a = self.conv2(seg_result)
		result_b = self.conv3(seg_result)

		return tf.math.add(tf.math.multiply(norm, tf.math.add(1, result_a)), result_b) """
		return self.conv3(features)