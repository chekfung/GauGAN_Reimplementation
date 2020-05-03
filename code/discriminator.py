import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose 
from tensorflow_addons.layers import InstanceNormalization
from code.spectral_norm import spectral_conv


# forward is call
# - it is feeding input through the layers

class Discriminator(Model):
    def __init__(self, segmap_filters, beta1=0.5, beta2=0.999, learning_rate=0.0004):
        super(Discriminator, self).__init__()
        # Padding, Stride, etc calculations
        KERNEL_SIZE = 4
        ALPHA_VAL = 0.2

        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, beta_1 = self.beta1, beta_2 = self.beta2)

        # Initial first block
        self.glorot = tf.keras.initializers.GlorotNormal()
        # filters=64, stride=2
        self.conv1 = tf.Variable(self.glorot(shape=[KERNEL_SIZE, KERNEL_SIZE, segmap_filters+3, 64]))
        self.bias1 = tf.Variable(self.glorot(shape=[64]))
        self.leaky1 = LeakyReLU(alpha=ALPHA_VAL)

        # Second block
        self.conv2 = tf.Variable(self.glorot(shape=[KERNEL_SIZE, KERNEL_SIZE, 64, 128]))
        self.bias2 = tf.Variable(self.glorot(shape=[128]))
        self.inorm1 = InstanceNormalization()
        self.leaky2 = LeakyReLU(alpha=ALPHA_VAL)

        # Third block
        self.conv3 = tf.Variable(self.glorot(shape=[KERNEL_SIZE, KERNEL_SIZE, 128, 256]))
        self.bias3 = tf.Variable(self.glorot(shape=[256]))
        self.inorm2 = InstanceNormalization()
        self.leaky3 = LeakyReLU(alpha=ALPHA_VAL)

        # Fourth block
        self.conv4 = tf.Variable(self.glorot(shape=[KERNEL_SIZE, KERNEL_SIZE, 256, 512]))
        self.bias4 = tf.Variable(self.glorot(shape=[512]))
        self.inorm3 = InstanceNormalization()
        self.leaky4 = LeakyReLU(alpha=ALPHA_VAL)

        # Final Convolutional Layer, as like PatchGAN implementation
        self.conv5 = tf.Variable(self.glorot(shape=[KERNEL_SIZE, KERNEL_SIZE, 512, 1]))
        self.bias5 = tf.Variable(self.glorot(shape=[1]))

        # In weird pytorch code 
        self.inorm4 = InstanceNormalization()
        self.leaky5 = LeakyReLU(alpha=ALPHA_VAL)

        self.bce = tf.keras.losses.BinaryCrossentropy()

    def call(self, inputs, segmaps):
        x = tf.concat([segmaps, inputs], axis=-1)
        
        # First layer
        x = spectral_conv(inputs=x, weight=self.conv1, stride=2, bias=self.bias1)
        x = self.leaky1(x)

        # Second Layer
        x = spectral_conv(inputs=x, weight=self.conv2, stride=2, bias=self.bias2)
        x = self.inorm1(x)
        x = self.leaky2(x)

        # Third Layer
        x = spectral_conv(inputs=x, weight=self.conv3, stride=2, bias=self.bias3)
        x = self.inorm2(x)
        x = self.leaky3(x)

        # Fourth layer
        x = spectral_conv(inputs=x, weight=self.conv4, stride=2, bias=self.bias4)
        x = self.inorm3(x)
        x = self.leaky4(x)

        # Final layer
        #x = self.leaky5(self.inorm4(x))
        x = spectral_conv(inputs=x, weight=self.conv5, stride=1, bias=self.bias5)

        return x

    """
    Paper concatenates fake and real images because in Batch Normalization, 
    concatenating "avoids disparate statistics in fake and real images". We have
    opted to skip this and return if we have time
    """
    def loss(self, real_output, fake_output):
        # Hinge loss from pytorch implementation
        real_loss = tf.math.multiply(-1, tf.reduce_mean(tf.minimum(tf.math.subtract(real_output, 1), 0)))
        fake_loss = tf.math.multiply(-1, tf.reduce_mean(tf.minimum(tf.math.multiply(-1, tf.math.subtract(fake_output, 1)), 0)))

        # NOTE: THIS INITIALLY HAD DIVISION BY 2. GOT RID OF IT SO THAT REACHES 0 LATER.
        return tf.divide(tf.reduce_mean(tf.math.add(real_loss, fake_loss)), 2)
        
        # BCE loss 
        """ loss1 = self.bce(tf.ones_like(real_output), real_output)
        loss2 = self.bce(tf.zeros_like(fake_output), fake_output)
        return tf.math.add(loss1, loss2)  """

        # Least Squares loss
        # return tf.math.multiply(0.5,tf.math.add((tf.reduce_mean((real_output - 1)**2), \
        #     tf.reduce_mean(fake_output**2))))
        

