from code.spadelayer import SpadeLayer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Layer, ReLU
from code.spectral_norm import spectral_conv

class SpadeBlock(Layer): 
	def __init__(self, fin, fout, segmap_filters, use_bias=True, use_spectral=True, skip=False): 
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
		self.glorot = tf.keras.initializers.GlorotNormal()
		
		# filters out = fmiddle, kernel=3, strides=1
		self.conv0 = tf.Variable(self.glorot(shape=[3,3,fin,fmiddle]))
		self.bias0 = tf.Variable(self.glorot(shape=[fmiddle]))
		# filters out = fout, kernel=3, strides=1
		self.conv1 = tf.Variable(self.glorot(shape=[3,3,fmiddle,fout]))
		self.bias1 = tf.Variable(self.glorot(shape=[fout]))
		# filters out = fout, kernel=1, strides=1
		if self.learned_shortcut: 
			self.conv_s = tf.Variable(self.glorot(shape=[1,1,fin,fout]))

		self.spade0 = SpadeLayer(in_channels=segmap_filters, out_channels=fin)
		self.spade1 = SpadeLayer(in_channels=segmap_filters, out_channels=fmiddle)
		#self.spade_s = SpadeLayer(out_channels=fin) #comment 
		if self.learned_shortcut: 
			self.spade_s = SpadeLayer(in_channels=segmap_filters, out_channels=fin)
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
			x = spectral_conv(inputs=x, weight=self.conv0, stride=1, bias=self.bias0)
			x = self.relu(self.spade1(x, segmap))
			x = spectral_conv(inputs=x, weight=self.conv1, stride=1, bias=self.bias1)

			if self.learned_shortcut: 
				skip = self.relu(self.spade_s(skip, segmap))
				skip = spectral_conv(inputs=skip, weight=self.conv_s, stride=1, use_bias=False)
		else: 
			skip = features
			x = self.relu(self.spade0(features, segmap))
			x = tf.nn.conv2d(x, self.conv0, [1,1,1,1], "SAME")
			x = tf.nn.bias_add(x, self.bias0)
			x = self.relu(self.spade1(x, segmap))
			x = tf.nn.conv2d(x, self.conv1, [1,1,1,1], "SAME")
			x = tf.nn.bias_add(x, self.bias1)

			if self.learned_shortcut: 
				skip = self.relu(self.spade_s(skip, segmap))
				skip = tf.nn.conv2d(skip, self.conv_s, [1,1,1,1], "SAME")

		return tf.math.add(skip, x)