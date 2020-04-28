from spadelayer import SpadeLayer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Layer

class SpadeBlock(Layer): 
	def __init__(self, fin, fout, use_bias=True, use_spectral=False): 
		super(SpadeBlock, self).__init__()
		self.use_spectral = use_spectral 

		self.learned_shortcut = (fin != fout)
		fmiddle = min(fin, fout)

		self.conv0 = Conv2D(filters=fmiddle, kernel_size=3, strides=1, padding="SAME", use_bias=use_bias)
		self.conv1 = Conv2D(filters=fout, kernel_size=3, strides=1, padding="SAME", use_bias=use_bias)
		if self.learned_shortcut: 
			self.conv_s = Conv2D(filters=fout, kernel_size=1, strides=1, padding="SAME", use_bias=False)
		
		self.spade0 = SpadeLayer(out_channels=fin)
		self.spade1 = SpadeLayer(out_channels=fmiddle)
		if self.learned_shortcut: 
			self.spade_s = SpadeLayer(out_channels=fin)

	def call(self, features, segmap): 
		if self.use_spectral: 
			skip_features = self.spectral_norm(w=self.shortcut(features, segmap))
			dx = self.spectral_norm(w=self.conv0(self.actvn(self.spade0(features, segmap))))
			dx = self.spectral_norm(w=self.conv1(self.actvn(self.spade1(dx, segmap))))
			out = skip_features + dx
		else: 
			skip_features = self.shortcut(features, segmap)
			dx = self.conv0(self.actvn(self.spade0(features, segmap)))
			dx = self.conv1(self.actvn(self.spade1(dx, segmap)))
			out = skip_features + dx
		return out
	
	def shortcut(self, features, segmap): 
		if self.learned_shortcut: 
			x_s = self.conv_s(self.spade_s(features, segmap))
		else: 
			x_s = features
		return x_s
	
	def actvn(self, x): 
		return LeakyReLU(x, alpha=0.2)

	"""
	This spectral_norm implementation was taken from https://github.com/taki0112/Spectral_Normalization-Tensorflow
	"""
	def spectral_norm(self, w, iteration=1):
		w_shape = w.shape.as_list()
		w = tf.reshape(w, [-1, w_shape[-1]])

		u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

		u_hat = u
		v_hat = None
		for i in range(iteration):
			"""
			power iteration
			Usually iteration = 1 will be enough
			"""
			v_ = tf.matmul(u_hat, tf.transpose(w))
			v_hat = tf.nn.l2_normalize(v_)

			u_ = tf.matmul(v_hat, w)
			u_hat = tf.nn.l2_normalize(u_)

		u_hat = tf.stop_gradient(u_hat)
		v_hat = tf.stop_gradient(v_hat)

		sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

		with tf.control_dependencies([u.assign(u_hat)]):
			w_norm = w / sigma
			w_norm = tf.reshape(w_norm, w_shape)


		return w_norm