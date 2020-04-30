from code.spadelayer import SpadeLayer  
from code.spadeblock import SpadeBlock
from tensorflow.keras import Model
from test_layer import MyDenseLayer
import tensorflow as tf

class MOO(Model):
	def __init__(self, size1, size2):
		super(MOO, self).__init__()
		self.size2 = size2
		if size1 - size2 > 10: 
			self.size2 = 15
		self.size1 = size1
		self.layer = MyDenseLayer(self.size1)
		self.layer1 = SpadeLayer(self.size2)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.5, beta_2 = 0.999)


	def call(self, input, second):
		x = self.layer(input)
		x = self.layer1(second, second)
		return x
		

random = tf.random.normal((4, 4))
random2 = tf.random.normal((4,4,4,20))
##layers_of_onion = 10
#onion_layer = SpadeLayer(layers_of_onion)





moo_obj = MOO(10, 20)

# Do stuff
with tf.GradientTape() as tape: 
	output = moo_obj(random2, random2)
	#output = onion_layer(random, random)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, tf.zeros(output.shape)))

	grads = tape.gradient(loss, moo_obj.trainable_variables)

moo_obj.optimizer.apply_gradients(zip(grads, moo_obj.trainable_variables))

print(moo_obj.trainable_variables)
print("Ouch")

