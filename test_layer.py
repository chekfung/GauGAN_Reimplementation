import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Layer, Dense

class MyDenseLayer(tf.keras.Model):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs
    self.fc = tf.keras.layers.Dense(num_outputs)

  def call(self, input):
    return self.fc(input)

#   def compute_output_shape(self, input_shape):
#     shape = tf.TensorShape(input_shape).as_list()
#     shape[-1] = self.num_outputs
#     return tf.TensorShape(shape)