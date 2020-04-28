from spadelayer import SpadeLayer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Model, Flatten

class Encoder(Model): 

	def __init__(self): 
		super(Encoder, self).__init__()
		self.model = tf.keras.Sequential()
		self.model.add(Conv2D(filters=64, kernel__size=3, stride=2, padding="SAME", use_bias=True))
		self.model.add(BatchNormalization()) # NOTE: MAY WANT TO SWITCH TO INSTANCE NORMALIZTION
		self.model.add(LeakyReLU(alpha=0.2))
		self.model.add(Conv2D(filters=128, kernel__size=3, stride=2, padding="SAME", use_bias=True))
		self.model.add(BatchNormalization()) # NOTE: MAY WANT TO SWITCH TO INSTANCE NORMALIZTION
		self.model.add(LeakyReLU(alpha=0.2))
		self.model.add(Conv2D(filters=256, kernel__size=3, stride=2, padding="SAME", use_bias=True))
		self.model.add(BatchNormalization()) # NOTE: MAY WANT TO SWITCH TO INSTANCE NORMALIZTION
		self.model.add(LeakyReLU(alpha=0.2))
		self.model.add(Conv2D(filters=512, kernel__size=3, stride=2, padding="SAME", use_bias=True))
		self.model.add(BatchNormalization()) # NOTE: MAY WANT TO SWITCH TO INSTANCE NORMALIZTION
		self.model.add(LeakyReLU(alpha=0.2))
		self.model.add(Conv2D(filters=512, kernel__size=3, stride=2, padding="SAME", use_bias=True))
		self.model.add(BatchNormalization()) # NOTE: MAY WANT TO SWITCH TO INSTANCE NORMALIZTION
		self.model.add(LeakyReLU(alpha=0.2))
		self.model.add(Conv2D(filters=512, kernel__size=3, stride=2, padding="SAME", use_bias=True))
		self.model.add(BatchNormalization()) # NOTE: MAY WANT TO SWITCH TO INSTANCE NORMALIZTION
		self.model.add(LeakyReLU(alpha=0.2))
		self.model.add(Flatten())
		self.mu = Dense(256, use_bias=True)
		self.var = Dense(256, use_bias=True)
		self.beta1 = 0.5
		self.learn_rate = 0.0002
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learn_rate, beta_1=self.beta1)

	def call(self, input): 
		intermediate_result = self.model(input)
		mu = self.mu(intermediate_result)
		var = self.var(intermediate_result)

		return mu, var

	