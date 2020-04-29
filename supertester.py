from code.spadelayer import SpadeLayer  
from code.spadeblock import SpadeBlock
from tensorflow.keras import Model
from test_layer import MyDenseLayer
import tensorflow as tf

class MOO(Model):
    def __init__(self):
        super(MOO, self).__init__()
        self.layer = MyDenseLayer(10)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.5, beta_2 = 0.999)


    def call(self, input):
        return self.layer(input)
        

random = tf.random.normal((4, 4))

##layers_of_onion = 10
#onion_layer = SpadeLayer(layers_of_onion)




moo_obj = MyDenseLayer(10)

# Do stuff
with tf.GradientTape() as tape: 
    output = moo_obj(random)
    #output = onion_layer(random, random)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.ones([3,1]), tf.zeros([3,1])))

grads = tape.gradient(loss, moo_obj.trainable_variables)
moo_obj.optimizer.apply_gradients(zip(grads, moo_obj.trainable_variables))

print(moo_obj.trainable_variables)
print("Ouch")

