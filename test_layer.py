import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Layer, Dense

class MyDenseLayer(tf.keras.Model):
	def __init__(self, num_outputs):
		super(MyDenseLayer, self).__init__()
		self.num_outputs = num_outputs
		self.fc = tf.keras.layers.Dense(num_outputs)
		self.optimizer = self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.5, beta_2 = 0.999)
	
	def call(self, input):
		return self.fc(input)
    
    def build(self, input_shape):
        self.fc.build(input_shape)
        self._trainable_weights = self.fc.trainable_weights
        super(MyDenselayer, self).build(input_shape)
		
	
	def compute_output_shape(self, input_shape):
		shape = tf.TensorShape(input_shape).as_list()
		shape[-1] = self.num_outputs
		return tf.TensorShape(shape)


# class mylayer(tf.keras.layers.Layer):
#     def __init__(self, num_outputs, num_outputs2):
#         self.num_outputs = num_outputs
#         super(mylayer, self).__init__()

#     def build(self, input_shape):
#         self.fc = tf.keras.layers.Dense(self.num_outputs)
#         self.fc.build(input_shape)
#         self._trainable_weights = self.fc.trainable_weights
#         super(mylayer, self).build(input_shape)

#     def call(self, input):
#         return self.fc(input)