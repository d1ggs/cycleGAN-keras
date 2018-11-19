import tensorflow as tf
from keras.layers import Layer, InputSpec


from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import ZeroPadding2D

import numpy as np

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        print (input_shape)
        return (input_shape[0], input_shape[1], input_shape[2] + 2 * self.padding[0], input_shape[3] + 2 * self.padding[1])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [0,0] ,[h_pad,h_pad], [w_pad,w_pad]], 'REFLECT')