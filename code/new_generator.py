import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Layer, ReLU
from spadeblock import SpadeBlock

class SPADEGenerator(tf.keras.Model):
	def __init__(self, beta1=0.5, beta2=0.999, learning_rate=0.0001, batch_size=32, z_dim=64, \
		img_w=40, img_h=30):
		super(SPADEGenerator, self).__init__()
		self.spade_layers = []
		self.spade_blocks = []


    # ======================================================================= #
    # Spade layer functions
	def spade_layer(self, out_channels, use_bias=True, hidden_channels=128): 
        layer = [
            BatchNormalization(), 
            Conv2D(filters=hidden_channels, kernel_size=5, strides=1, padding="SAME", use_bias=use_bias), 
            ReLU(), 
            Conv2D(filters=out_channels, kernel_size=5, strides=1, padding="SAME", use_bias=use_bias),
            Conv2D(filters=out_channels, kernel_size=5, strides=1, padding="SAME", use_bias=use_bias)
        ]
        self.spade_layers.append(layer)

	# call one spade layer 
	def call_one_spade_layer(self, layer_list, features, segmap): 
        norm = layer_list[0](features)
        _, x_h, x_w, _ = list(norm.shape)
        segmap_resized = tf.image.resize(segmap, size=(x_h, x_w), method="nearest")
        seg_result = layer_list[2](layer_list[1](segmap_resized))
        result_a = layer_list[3](seg_result)
        result_b = layer_list[4](seg_result)

        return norm * (1 + result_a) + result_b
    
    # ======================================================================= #

	def spade_block(self, fin, fout, use_bias=True, use_spectral=False):
        fmiddle = min(fin, fout)
        learned_shortcut = fin != fout

		layer = [
            Conv2D(filters=fmiddle, kernel_size=3, strides=1, padding="SAME", use_bias=use_bias),
            Conv2D(filters=fout, kernel_size=3, strides=1, padding="SAME", use_bias=use_bias),
        ]

        if learned_shortcut:
            layer.append(Conv2D(filters=fout, kernel_size=1, strides=1, padding="SAME", use_bias=False))




        # FIXME: Convert this to using the spade layer functions
        self.spade0 = SpadeLayer(out_channels=fin)
		self.spade1 = SpadeLayer(out_channels=fmiddle)
		if self.learned_shortcut: 
			self.spade_s = SpadeLayer(out_channels=fin)


        # Now to do with the spade layers  
        
        self.spade_blocks.append(layer)
		
		def shortcut(features, segmap): 
		if learned_shortcut: 
			x_s = conv_s(spade_s(features, segmap))
		else: 
			x_s = features
		return x_s
	
	def actvn(self, x): 
		return LeakyReLU(x, alpha=0.2)



	def __init__(self, fin, fout, use_bias=True, use_spectral=False): 

		#spade block
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


		# Generator 
		self.beta1 = beta1
		self.beta2 = beta2
		self.learning_rate = learning_rate
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, beta_1 = self.beta1, beta_2 = self.beta2)
		self.batch_size = batch_size
		self.num_spade_layers = 7
		self.spade_layers = []
		self.num_channels = z_dim
		self.image_width = img_w
		self.image_height = img_h
		self.dense = tf.keras.layers.Dense(self.image_height*self.image_width*self.num_channels)
		self.spade_layers.append(SpadeBlock(z_dim, int(z_dim / 2)))
		self.spade_layers.append(SpadeBlock(int(z_dim / 2), int(z_dim / 4)))
		self.spade_layers.append(SpadeBlock(int(z_dim / 4), int(z_dim / 8)))
		# May need to change this 64.
		self.conv_layer = tf.keras.layers.Conv2D(64, (3,3), activation='tanh')
		
		
		
		
		# Spade Block 1



		


		# Spade Block 2







		# Spade Block 3


	


	
	
	

	#generator call, loss, upsample
	def call(self, images):
		batch_size = np.shape(images)[0]
		noise = np.random.normal(loc=0, scale=1, size=(batch_size, self.num_channels))
		result_dense = self.dense(noise)
		reshaped = tf.reshape(result_dense, [batch_size, self.image_width, self.image_height, self.num_channels])
		result = self.spade_layers[0](reshaped, images)
		for layer in self.spade_layers[1:]:
			result = upsample(result)
			result = layer(result, images)
		result = tf.nn.leaky_relu(result)
		result = self.conv_layer(result)
		return result
	
	def loss(self,fake_logits):
		# Only hinge loss for now--can add extra losses later
		return tf.keras.losses.hinge(tf.zeros_like(fake_logits), fake_logits)
	
	def upsample(batch_inputs):
		return tf.image.resize(batch_inputs, [2*np.shape(batch_inputs)[1], 2*np.shape(batch_inputs)[2]])
	
	




	


class SpadeBlock(tf.keras.Model): 
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

	