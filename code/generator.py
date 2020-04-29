import numpy as np
import tensorflow as tf
from code.spadeblock import SpadeBlock
from tensorflow.keras.layers import UpSampling2D, LeakyReLU

class SPADEGenerator(tf.keras.Model):
    def __init__(self, beta1=0.5, beta2=0.999, learning_rate=0.0001, batch_size=32, z_dim=64, \
        img_w=40, img_h=30):
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
        #self.spade_layers.append(SpadeBlock(z_dim, z_dim))
        #self.spade_layers.append(SpadeBlock(z_dim, z_dim))
        #self.spade_layers.append(SpadeBlock(z_dim, z_dim))
        self.spade_layers.append(SpadeBlock(z_dim, int(z_dim / 2)))
        self.spade_layers.append(SpadeBlock(int(z_dim / 2), int(z_dim / 4)))
        self.spade_layers.append(SpadeBlock(int(z_dim / 4), int(z_dim / 8)))
        #self.spade_layers.append(SpadeBlock(int(z_dim / 8), int(z_dim / 16)))
        # May need to change this 64.
        self.conv_layer = tf.keras.layers.Conv2D(64, (3,3), activation='tanh')
        self.upsample = UpSampling2D()
        self.lrelu = LeakyReLU(alpha=0.2)
    
    def call(self, images, noise):
        result_dense = self.dense(noise)
        reshaped = tf.reshape(result_dense, [-1, self.image_width, self.image_height, self.num_channels])
        result = self.spade_layers[0](reshaped, images)
        for layer in self.spade_layers[1:]:
            result = self.upsample(result)
            result = layer(result, images)
        result = self.lrelu(result)
        result = self.conv_layer(result)
        return result
    
    def loss(self,fake_logits):
        # Only hinge loss for now--can add extra losses later
        return tf.keras.losses.hinge(tf.zeros_like(fake_logits), fake_logits)
    
"""     @tf.function
    def upsample(self, batch_inputs):
        return tf.image.resize(batch_inputs, [2*np.shape(batch_inputs)[1], 2*np.shape(batch_inputs)[2]]) """
