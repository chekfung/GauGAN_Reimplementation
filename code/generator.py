import numpy as np
import tensorflow as tf
from spadeblock import SpadeBlock



class SPADEGenerator(tf.keras.Model):
    def __init__(self, beta1=0.5, beta2=0.999, learning_rate=0.0001, batch_size=32, z_dim=1024, \
        img_w=160, img_h=120):
        super(SPADEGenerator, self).__init__()
        
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
        self.spade_layers.append(SpadeBlock(1024, 1024))
        self.spade_layers.append(SpadeBlock(1024, 1024))
        self.spade_layers.append(SpadeBlock(1024, 1024))
        self.spade_layers.append(SpadeBlock(1024, 512))
        self.spade_layers.append(SpadeBlock(512, 256))
        self.spade_layers.append(SpadeBlock(256, 128))
        self.spade_layers.append(SpadeBlock(128, 64))
        self.conv_layer = tf.keras.layers.Conv2D(64, (3,3), activation='tanh')
        
        self.adversial_weight = 1
    def call(self, images):
        batch_size = np.shape(images)[0]
        noise = np.random.normal((batch_size, 1024))
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