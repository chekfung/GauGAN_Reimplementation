import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, 
from tensorflow_addons.layers import InstanceNormalization


# forward is call
# - it is feeding input through the layers

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Padding, Stride, etc calculations
        KERNEL_SIZE = 4
        pad_size = int(np.ceil((KERNEL_SIZE - 1) / 2))
        ALPHA_VAL = 0.2

        self.model = tf.keras.Sequential()

        # Initial first block
        self.model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, strides=2, padding=pad_size))
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Second block
        self.model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=2, padding=pad_size))
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Third block
        self.model.add(Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=2, padding=pad_size))
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Fourth block
        self.model.add(Conv2D(filters=512, kernel_size=KERNEL_SIZE, strides=1, padding=pad_size))
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Final Convolutional Layer, as like PatchGAN implementation
        self.model.add(Conv2D(filters=1, kernel_size=KERNEL_SIZE, strides=1, paddings=pad_size))


    @tf.function
    def call(self, inputs, segmaps):
        x = tf.concat([segmaps, inputs], axis=-1)
        return self.model(x)

    """
    Paper concatenates fake and real images because in Batch Normalization, 
    concatenating "avoids disparate statistics in fake and real images". We have
    opted to skip this and return if we have time
    """
    def discriminator_loss(fake_output, real_output):
        real_loss = -tf.reduce_mean(tf.minimum(real - 1, 0))
        fake_loss = -tf.reduce_mean(tf.minimum(-fake - 1, 0))

        return tf.reduce_mean(real_loss + fake_loss)