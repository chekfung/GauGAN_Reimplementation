from code.spadelayer import SpadeLayer  
from tensorflow.keras import Model
import tensorflow as tf

class MOO(Model):
    def __init__(self):
        super(MOO, self).__init__()
        self.layer = SpadeLayer(10)

    def call(self, img, segmap):
        return self.layer(img, segmap)
        

random = tf.random.normal((4, 4, 4, 4))

layers_of_onion = 10
onion_layer = SpadeLayer(layers_of_onion)
moo_obj = MOO()

# Do stuff
with tf.GradientTape() as tape: 
    output = moo_obj(random,random)
    #output = onion_layer(random, random)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.ones([3,1]), tf.zeros([3,1])))

grads = tape.gradient(loss, moo_obj.trainable_variables)

print(moo_obj.trainable_variables)

