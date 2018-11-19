from keras.layers import Conv2D, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import add
from keras import initializers
from keras_contrib.layers import InstanceNormalization
'''
Keras Customizable Residual Unit

This is a simplified implementation of the basic (no bottlenecks) full pre-activation residual unit from He, K., Zhang, X., Ren, S., Sun, J., "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027v2).
'''
conv_init = initializers.RandomNormal(0, 0.02)  # for convolution kernel
gamma_init = initializers.RandomNormal(1., 0.02)  # for batch normalization

def conv_block(feat_maps_out, prev):
    prev = InstanceNormalization(gamma_initializer=gamma_init)(prev, training=1)  # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)  # possibile migliore risultato con ReLU?
    prev = Conv2D(feat_maps_out, (3, 3), padding='same',
                  kernel_initializer=conv_init)(prev)
    prev = InstanceNormalization(gamma_initializer=gamma_init)(prev, training=1)  # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, (3, 3), padding='same',
                  kernel_initializer=conv_init)(prev)
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Conv2D(feat_maps_out, (1, 1), padding='same',
                      kernel_initializer=conv_init)(prev)
    return prev 


def Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks

    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    #print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
    return add([skip, conv]) # the residual connection

