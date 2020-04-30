import numpy as np
import tensorflow as tf
from code.spadeblock import SpadeBlock
from tensorflow.keras.layers import UpSampling2D, LeakyReLU, Conv2D

class SPADEGenerator(tf.keras.Model):
    def __init__(self, beta1=0.5, beta2=0.999, learning_rate=0.0001, batch_size=16, z_dim=64, \
        img_w=40, img_h=30):
        super(SPADEGenerator, self).__init__()
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, beta_1 = self.beta1, beta_2 = self.beta2)
        self.batch_size = batch_size
        self.num_channels = z_dim
        self.upsample_count = 5
        self.img_w = img_w
        self.img_h = img_h

        self.sw, self.sh = self.compute_latent_vector_size()
        self.fc = Conv2D(16 * z_dim, kernel_size=3, strides=1, padding="SAME", use_bias=True, dtype=tf.float32)
        #self.dense = tf.keras.layers.Dense(self.sh*self.sw*self.num_channels, dtype=tf.float32)
        #self.dense = tf.keras.layers.Dense(16384)

        # For now, I just make this NF like the code
        nf = z_dim

        # SPADE LAYERS
        self.spade_layers0 = SpadeBlock(16 * nf, 16 * nf)
        self.spade_layers1 = SpadeBlock(16 * nf, 16 * nf)
        self.spade_layers2 = SpadeBlock(16 * nf, 16 * nf)
        self.spade_layers3 = SpadeBlock(16 * nf, 8 * nf)
        self.spade_layers4 = SpadeBlock(8 * nf, 4 * nf)
        self.spade_layers5 = SpadeBlock(4 * nf, 2 * nf)
        self.spade_layers6 = SpadeBlock(2 * nf, 1 * nf)

        self.conv_layer = tf.keras.layers.Conv2D(3, (3,3), padding="SAME", activation='tanh', dtype=tf.float32)

        # Unsample layer by 2
        self.upsample = UpSampling2D()


        self.lrelu = LeakyReLU(alpha=0.2)
        self.bce = tf.keras.losses.BinaryCrossentropy()
    
    def call(self, segs):
        #result_dense = self.dense(noise)
        #reshaped = tf.reshape(result_dense, [-1, self.image_width, self.image_height, self.num_channels])
        #reshaped = tf.reshape(result_dense, [segs.shape[0], -1, 4, 4])
        result = tf.image.resize(segs, size=(self.sh, self.sw), method="nearest")
        result = self.fc(result)

        # Start doing spade layers
        result = self.spade_layers0(result, segs)
        result = self.upsample(result)
        print("layer0: ", result.shape)
        # Middle layers
        result = self.spade_layers1(result, segs)
        print("layer1: ", result.shape)
        result = self.spade_layers2(result, segs)
        print("layer2:", result.shape)

        # Rest of the layers
        result = self.upsample(result)
        print("Unsampled layer2:", result.shape)
        result = self.spade_layers3(result, segs)
        print("layer3: ", result.shape)
        
        result = self.upsample(result)
        result = self.spade_layers4(result, segs)
        print("layer4:", result.shape)

        result = self.upsample(result)
        result = self.spade_layers5(result, segs)
        print("layer5: ", result.shape)

        result = self.upsample(result)
        result = self.spade_layers6(result, segs)
        print("Layer6: ", result.shape)

        # Take activation function plus final convolution layer in generator
        result = self.lrelu(result)
        result = self.conv_layer(result)

        return result
    

    def compute_latent_vector_size(self):
        sw = self.img_w // (2**self.upsample_count)
        sh = round(sw / (4/3))
        return sw, sh

    
    def loss(self,fake_logits):
        # Only hinge loss for now--can add extra losses later
        #return tf.keras.losses.hinge(tf.zeros_like(fake_logits), fake_logits)
        return self.bce(tf.ones_like(fake_logits), fake_logits)
