from code.spadelayer import SpadeLayer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Layer, ReLU
from code.spectral import spectral_norm

class SpadeBlock(Layer): 
	def __init__(self, fin, fout, use_bias=True, use_spectral=True, skip=False): 
		super(SpadeBlock, self).__init__()
		#self.use_spectral = use_spectral 

		""" self.skip = skip
		
		if self.skip: 
			self.spade1 = SpadeLayer(2*k)
			self.conv1 = Conv2D(k, kernel_size=3, padding="SAME", use_bias=use_bias, dtype=tf.float32)
			self.spade_skip = SpadeLayer(k)
			self.conv_skip = Conv2D(k, kernel_size=3, padding="SAME", use_bias=use_bias, dtype=tf.float32)
		else: 
			self.spade1 = SpadeLayer(k)
			self.conv1 = Conv2D(k, kernel_size=3, padding="SAME", use_bias=use_bias, dtype=tf.float32)

		self.spade2 = SpadeLayer(k)
		self.conv2 = Conv2D(k, kernel_size=3, padding="SAME", use_bias=use_bias, dtype=tf.float32)
		self.relu = ReLU() """
		self.use_spectral = use_spectral
		self.learned_shortcut = (fin != fout)
		fmiddle = min(fin, fout)
		
		self.conv0 = Conv2D(filters=fmiddle, kernel_size=3, strides=1, padding="SAME", \
			use_bias=use_bias, dtype=tf.float32, kernel_initializer=tf.keras.initializers.GlorotNormal())
		self.conv1 = Conv2D(filters=fout, kernel_size=3, strides=1, padding="SAME", \
			use_bias=use_bias, dtype=tf.float32, kernel_initializer=tf.keras.initializers.GlorotNormal())
		#self.conv_s = Conv2D(filters=fout, kernel_size=1, strides=1, padding="SAME", use_bias=False) # comment
		if self.learned_shortcut: 
			self.conv_s = Conv2D(filters=fout, kernel_size=1, strides=1, padding="SAME", \
				use_bias=False, dtype=tf.float32, kernel_initializer=tf.keras.initializers.GlorotNormal())
		
		self.spade0 = SpadeLayer(out_channels=fin)
		self.spade1 = SpadeLayer(out_channels=fmiddle)
		#self.spade_s = SpadeLayer(out_channels=fin) #comment 
		if self.learned_shortcut: 
			self.spade_s = SpadeLayer(out_channels=fin)
		self.relu = ReLU()

	def call(self, features, segmap): 
		""" skip_features = self.shortcut(features, segmap)
		#skip_features = self.conv_s(self.spade_s(features, segmap))
		dx = self.conv0(self.lrelu1(self.spade0(features, segmap)))
		dx = self.conv1(self.lrelu2(self.spade1(dx, segmap)))
		out = tf.math.add(skip_features, dx) """
		if self.use_spectral: 
			skip = features
			x = self.relu(self.spade0(features, segmap))
			x = spectral_norm(self.conv0(x))
			x = self.relu(self.spade1(x, segmap))
			x = spectral_norm(self.conv1(x))

			if self.learned_shortcut: 
				skip = self.relu(self.spade_s(skip, segmap))
				skip = spectral_norm(self.conv_s(skip))
		else: 
			skip = features
			x = self.relu(self.spade0(features, segmap))
			x = self.conv0(x)
			x = self.relu(self.spade1(x, segmap))
			x = self.conv1(x)

			if self.learned_shortcut: 
				skip = self.relu(self.spade_s(skip, segmap))
				skip = self.conv_s(skip)

		return tf.math.add(skip, x)
	
	def shortcut(self, features, segmap): 
		if self.learned_shortcut: 
			x_s = self.conv_s(self.spade_s(features, segmap))
		else: 
			x_s = features
		return x_s