from code.spadelayer import SpadeLayer  
import tensorflow as tf
#import 

random = tf.random.normal((4, 4, 4, 4))

layers_of_onion = 10
onion_layer = SpadeLayer(layers_of_onion)

# Do stuff
with tf.GradientTape() as tape: 
	output = onion_layer(random, random)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.ones([3,1]), tf.zeros([3,1])))

grads = tape.gradient(loss, onion_layer.trainable_variables)

print(onion_layer.trainable_parameters)

