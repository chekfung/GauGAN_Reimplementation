import tensorflow as tf
from keras.applications.vgg19 import preprocess_input

class VGG(tf.keras.Model): 
	
	def __init__(self, trainable=False): 
		super(VGG, self).__init__(name="Vgg19")
		vgg_feats = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(128,96,3))

		vgg_feats.trainable = trainable
		
		vgg_feats = vgg_feats.layers

		self.section1 = tf.keras.Sequential()
		self.section2 = tf.keras.Sequential()
		self.section3 = tf.keras.Sequential()
		self.section4 = tf.keras.Sequential()
		self.section5 = tf.keras.Sequential()
		
		for i in range(1,2): 
			self.section1.add(vgg_feats[i])
		for i in range(2,5): 
			self.section2.add(vgg_feats[i])
		for i in range(5,8): 
			self.section3.add(vgg_feats[i])
		for i in range(8,13): 
			self.section4.add(vgg_feats[i])
		for i in range(13,18): 
			self.section5.add(vgg_feats[i])
		
		# According to model provided in NVLabs code in architecture.py
		# for i in range(1,2): 
		# 	self.section1.add(vgg_feats[i])
		# for i in range(2,7): 
		# 	self.section2.add(vgg_feats[i])
		# for i in range(7,12): 
		# 	self.section3.add(vgg_feats[i])
		# for i in range(12,21): 
		# 	self.section4.add(vgg_feats[i])
		# for i in range(21,30): 
		# 	print(i)
		# 	self.section5.add(vgg_feats[i])


	def call(self, inputs): 
		out1 = self.section1(inputs)
		out2 = self.section2(out1)
		out3 = self.section3(out2)
		out4 = self.section4(out3)
		out5 = self.section5(out4)

		self.section1.summary()
		self.section2.summary()
		self.section3.summary()
		self.section4.summary()
		self.section5.summary()
		return out1, out2, out3, out4, out5


class VGG_Loss(tf.keras.Model): 

	def __init__(self): 
		super(VGG_Loss, self).__init__(name="Vgg_Loss")
		self.vgg = VGG()
		self.loss_function = tf.keras.losses.MeanAbsoluteError()
		self.weighting = [1/32, 1/16, 1/8, 1/4, 1]

	def call(self, fake, real): 
		fake_vgg, real_vgg = self.vgg(fake), self.vgg(real)
		loss = 0
		for i in range(len(fake_vgg)): 
			loss += self.weighting[i] * self.loss_function(fake_vgg[i], real_vgg[i])

		return loss
