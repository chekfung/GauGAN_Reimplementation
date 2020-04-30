import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose 
from tensorflow_addons.layers import InstanceNormalization


# forward is call
# - it is feeding input through the layers

class Discriminator(Model):
    def __init__(self, beta1=0.5, beta2=0.999, learning_rate=0.0004):
        super(Discriminator, self).__init__()
        # Padding, Stride, etc calculations
        KERNEL_SIZE = 4
        ALPHA_VAL = 0.2

        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, beta_1 = self.beta1, beta_2 = self.beta2)
        self.model = tf.keras.Sequential()

        # Initial first block
        self.model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, strides=2, padding="SAME"))
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Second block
        self.model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=2, padding="SAME"))
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Third block
        self.model.add(Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=2, padding="SAME"))
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Fourth block
        self.model.add(Conv2D(filters=512, kernel_size=KERNEL_SIZE, strides=1, padding="SAME"))
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Final Convolutional Layer, as like PatchGAN implementation
        self.model.add(Conv2D(filters=1, kernel_size=KERNEL_SIZE, strides=1, padding="SAME"))

        self.bce = tf.keras.losses.BinaryCrossentropy()


    @tf.function
    def call(self, inputs, segmaps):
        x = tf.concat([segmaps, inputs], axis=-1)
        return self.model(x)

    """
    Paper concatenates fake and real images because in Batch Normalization, 
    concatenating "avoids disparate statistics in fake and real images". We have
    opted to skip this and return if we have time
    """
    def loss(self, real_output, fake_output):
        """ real_loss = tf.math.multiply(-1, tf.reduce_mean(tf.minimum(tf.math.subtract(real_output, 1), 0)))
        fake_loss = tf.math.multiply(-1, tf.reduce_mean(tf.minimum(tf.math.multiply(-1, tf.math.subtract(fake_output, 1)), 0)))

        return tf.reduce_mean(tf.math.add(real_loss, fake_loss)) """
        loss1 = self.bce(tf.ones_like(real_output), real_output)
        loss2 = self.bce(tf.zeros_like(fake_output), fake_output)
        return tf.math.add(loss1, loss2)
