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
        self.model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, strides=2, padding=pad_size)
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Second block
        self.model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=2, padding=pad_size)
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Third block
        self.model.add(Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=2, padding=pad_size)
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Fourth block
        self.model.add(Conv2D(filters=512, kernel_size=KERNEL_SIZE, strides=1, padding=pad_size)
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(alpha=ALPHA_VAL))

        # Final Convolutional Layer, as like PatchGAN implementation
        self.model.add(Conv2D(filters=1, kernel_size=KERNEL_SIZE, strides=1, paddings=pad_size))


	@tf.function
    # I am pretty sure that this is wrong. Talk to Jeremy to determine how to fix it.
	def call(self, inputs, segmaps):
        # FIXME: Need to ask someone if this actually works as I believe it does.
        x = tf.concat([segmaps, inputs], axis=-1)
		return self.model(x)

    def discriminator_loss(segmap, generated, real):
        # I have no idea how to do the discriminator loss


    # NOTE: The loss function for the discriminator is going to be in the 
    # loss.py file of the model. 

    # Will talk with Jeremy to see if that makes sense. If not, I will 
    # just write it in here and have it here instead.