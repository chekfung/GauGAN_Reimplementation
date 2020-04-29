from code.spadelayer import SpadeLayer  
import tensorflow as tf
#import 

random = tf.random.normal((4, 4, 4, 4))

layers_of_onion = 10
onion_layer = SpadeLayer(layers_of_onion)

# Do stuff
with tf.GradientTape() as tape: 
	output = onion_layer(random, random)
	loss = 5

grads = tape.gradient(loss, onion_layer.trainable_variables)

print(onion_layer.trainable_parameters)

