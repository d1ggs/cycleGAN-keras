from keras import initializers
from keras.layers import Input, BatchNormalization, Activation, Lambda, ZeroPadding2D, Concatenate, Cropping2D, MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.utils.vis_utils import plot_model
from keras_contrib.layers import InstanceNormalization
import tensorflow as tf

from keras import backend as K

from residual import Residual
from unet_generator import generator_unet_deconv

import numpy as np

ngf = 32  # Number of filters in first layer of generator
ndf = 64  # Number of filters in first layer of discriminator
batch_size = 1  # batch_size
pool_size = 50  # pool_size
img_width = 256  # Input image will of width 256
img_height = 256  # Input image will be of height 256
img_depth = 3  # RGB format


conv_init = initializers.RandomNormal(0, 0.02)  # for convolution kernel
gamma_init = initializers.RandomNormal(1., 0.02)  # for batch normalization


# Custom Losses

def mae_loss(y_true, y_pred):

    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def reflectPadding(x, **kwargs):
    w_pad,h_pad = kwargs['padding']
    return tf.pad(x, [[0, 0], [0, 0], [h_pad, h_pad], [w_pad, w_pad]], 'REFLECT')


def calculatepadding(input_size, output_size, kernel_size, stride):
    return int(np.ceil((kernel_size + stride * (output_size -1) - input_size)/2))

def getpadding(w, h, kernel_size, stride):
    """Returns a tuple containing the padding value for the specified input dimensions
    for width and height of the layer. Note that w and h are tuples containing input and 
    output size for each dimension eg. (w_in, w_out)"""
    
    w_in, w_out = w
    h_in, h_out = h
    
    w_padding = calculatepadding(w_in, w_out, kernel_size, stride)
    if (w != h):
        h_padding = calculatepadding(h_in, h_out, kernel_size, stride)
    else:
        h_padding = w_padding
    return w_padding, h_padding



def ResNetGenerator(label, w, h):

    """Returns a model for the generator with two downsampling conv layers,
    9 residual blocks and two upsampling layers"""

    # Input
    inp = Input(shape=(3, h, w), name='Input')
    
    w_padding, h_padding = getpadding((w, w), (h, h), 7, 1)
    x = Lambda(reflectPadding, arguments={'padding': (h_padding, w_padding)})(inp)
    x = Conv2D(ngf, kernel_size=7, kernel_initializer=conv_init)(x)
    # bn0 = BatchNormalization()(cnv0)
    x = InstanceNormalization(gamma_initializer=gamma_init)(x, training=1)
    x = Activation('relu')(x)

    # Downsample
    w_padding, h_padding = getpadding((w, w/2), (h, h/2), 3, 2)
    x = Lambda(reflectPadding, arguments={'padding': (h_padding, w_padding)})(x)
    x = Conv2D(ngf*2, kernel_size=3, strides=2, kernel_initializer=conv_init)(x)
    x = InstanceNormalization(gamma_initializer=gamma_init)(x, training=1)
    x = Activation('relu')(x)

    w_padding, h_padding = getpadding((w/2, w/4), (h/2, h/4), 3, 2)
    x = Lambda(reflectPadding, arguments={'padding': (h_padding, w_padding)})(x)
    x = Conv2D(ngf*2*2, kernel_size=3, strides=2, kernel_initializer=conv_init)(x)
    x = InstanceNormalization(gamma_initializer=gamma_init)(x, training=1)
    x = Activation('relu')(x)

    # Residual blocks using constant filter number
    r1 = Residual(ngf*2*2, ngf*2*2, x)
    r2 = Residual(ngf*2*2, ngf*2*2, r1)
    r3 = Residual(ngf*2*2, ngf*2*2, r2)
    r4 = Residual(ngf*2*2, ngf*2*2, r3)
    r5 = Residual(ngf*2*2, ngf*2*2, r4)
    r6 = Residual(ngf*2*2, ngf*2*2, r5)

    # 9 residual blocks only for images >= 256*256, 6 blocks for 128*128 (as reported in cycleGAN paper)
    if w >= 256:
        r7 = Residual(ngf*2*2, ngf*2*2, r6)
        r8 = Residual(ngf*2*2, ngf*2*2, r7)
        r9 = Residual(ngf*2*2, ngf*2*2, r8)
        r_last = r9
    else:
        r_last = r6

    # Upsample
    x = InstanceNormalization(gamma_initializer=gamma_init)(r_last, training=1)
    x = Activation('relu')(x)
    x = Conv2DTranspose(ngf*2, kernel_size=3, strides=2, padding='same', kernel_initializer=conv_init)(x)
    x = InstanceNormalization(gamma_initializer=gamma_init)(x, training=1)
    x = Activation('relu')(x)

    x = Conv2DTranspose(ngf, kernel_size=3, strides=2, padding='same', kernel_initializer=conv_init)(x)
    x = InstanceNormalization(gamma_initializer=gamma_init)(x, training=1)
    x = Activation('relu')(x)

    # Output
    out = Conv2DTranspose(3, kernel_size=7, activation='tanh', padding='same', kernel_initializer=conv_init)(x)

    model = Model(inputs=inp, outputs=out, name='Generator'+label)

    return model


def PseudoUnet(label, w, h):

    """Returns a model for the generator with two downsampling conv layers,
    9 residual blocks and two upsampling layers"""

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    else:
        concat_axis = -1

    # Input
    inp = Input(shape=(3, h, w), name='Input')

    w_padding, h_padding = getpadding((w, w), (h, h), 7, 1)
    x = Lambda(reflectPadding, arguments={'padding': (h_padding, w_padding)})(inp)
    cnv1 = Conv2D(ngf, kernel_size=7, kernel_initializer=conv_init)(x)
    # bn0 = BatchNormalization()(cnv0)
    x = InstanceNormalization(gamma_initializer=gamma_init)(cnv1, training=1)
    x = Activation('relu')(x)

    # Downsample
    w_padding, h_padding = getpadding((w, w/2), (h, h/2), 3, 2)
    x = Lambda(reflectPadding, arguments={'padding': (h_padding, w_padding)})(x)
    cnv2 = Conv2D(ngf*2, kernel_size=3, strides=2, kernel_initializer=conv_init)(x)
    x = InstanceNormalization(gamma_initializer=gamma_init)(cnv2, training=1)
    x = Activation('relu')(x)

    w_padding, h_padding = getpadding((w/2, w/4), (h/2, h/4), 3, 2)
    x = Lambda(reflectPadding, arguments={'padding': (h_padding, w_padding)})(x)
    cnv3 = Conv2D(ngf*2*2, kernel_size=3, strides=2, kernel_initializer=conv_init)(x)
    x = InstanceNormalization(gamma_initializer=gamma_init)(cnv3, training=1)
    x = Activation('relu')(x)

    # Residual blocks using constant filter number
    r1 = Residual(ngf*2*2, ngf*2*2, x)
    r2 = Residual(ngf*2*2, ngf*2*2, r1)
    r3 = Residual(ngf*2*2, ngf*2*2, r2)
    r4 = Residual(ngf*2*2, ngf*2*2, r3)
    r5 = Residual(ngf*2*2, ngf*2*2, r4)
    r6 = Residual(ngf*2*2, ngf*2*2, r5)

    # 9 residual blocks only for images >= 256*256, 6 blocks for 128*128 (as reported in cycleGAN paper)
    if w >= 256:
        r7 = Residual(ngf*2*2, ngf*2*2, r6)
        r8 = Residual(ngf*2*2, ngf*2*2, r7)
        r9 = Residual(ngf*2*2, ngf*2*2, r8)
        r_last = r9
    else:
        r_last = r6

    # Upsample
    x = Concatenate(axis=concat_axis)([r_last, cnv3])
    x = InstanceNormalization(gamma_initializer=gamma_init)(x, training=1)
    x = Activation('relu')(x)
    x = Conv2DTranspose(ngf*2, kernel_size=3, strides=2, padding='same', kernel_initializer=conv_init)(x)
    x = InstanceNormalization(gamma_initializer=gamma_init)(x, training=1)
    x = Concatenate(axis=concat_axis)([x, cnv2])
    x = Activation('relu')(x)

    x = Conv2DTranspose(ngf, kernel_size=3, strides=2, padding='same', kernel_initializer=conv_init)(x)
    x = InstanceNormalization(gamma_initializer=gamma_init)(x, training=1)
    x = Concatenate(axis=concat_axis)([x, cnv1])
    x = Activation('relu')(x)

    # Output
    out = Conv2DTranspose(3, kernel_size=7, activation='tanh', padding='same', kernel_initializer=conv_init)(x)

    model = Model(inputs=inp, outputs=out, name='Generator'+label)

    return model


def PatchDiscriminator(name, w, h):

    """Returns a simple convolutional discriminator, implementing the PatchGAN 70X70 
    discriminator"""

    n_conv = 3

    inp = Input(shape=(3, h, w))
    x = inp

    for depth in range(n_conv):
        x = Conv2D(ndf*(2**depth), kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(x)
        if depth != 0:
            x = InstanceNormalization(gamma_initializer=gamma_init)(x, training=1)
        x = LeakyReLU(0.2)(x)

    # Last Conv
    x = ZeroPadding2D(1)(x)
    x = Conv2D(ndf * (2 ** n_conv), kernel_size=4, kernel_initializer=conv_init)(x)
    x = InstanceNormalization(gamma_initializer=gamma_init)(x, training=1)
    x = LeakyReLU(0.2)(x)

    # Decision layer
    x = ZeroPadding2D(1)(x)
    out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out, name="Discriminator"+name)

    return model


def components(w, h, pseudounet=False, unet=False, plot=True):
    """Returns all components for the cycleGAN architecture, using psudoUNet or
     UNet architecture for the generator if the respective parameters are set to True"""

    K.set_learning_phase(1)  # set learning phase

    disc_a = PatchDiscriminator("A", w, h)
    disc_b = PatchDiscriminator("B", w, h)
    if plot:
        plot_model(disc_a, to_file='./discriminator.png', show_shapes=True)

    if unet:
        print('\n Using UNet Generator model')
        gen_a2b = generator_unet_deconv((3, w, h), batch_size, model_name="_A2B")
        gen_b2a = generator_unet_deconv((3, w, h), batch_size, model_name="_B2A")
        if plot:
            plot_model(gen_a2b, to_file='./unet_generator.png', show_shapes=True)
    elif pseudounet:
        print('\n Using pseudoUNet Generator model')
        gen_a2b = PseudoUnet("_A2B", w, h)
        gen_b2a = PseudoUnet("_B2A", w, h)
        if plot:
            plot_model(gen_a2b, to_file='./pseudounet_generator.png', show_shapes=True)
    else:
        print('\n Using resNet Generator model')
        gen_a2b = ResNetGenerator("_A2B", w, h)
        gen_b2a = ResNetGenerator("_B2A", w, h)
        if plot:
            plot_model(gen_a2b, to_file='./resnet_generator.png', show_shapes=True)

    return disc_a, disc_b, gen_a2b, gen_b2a
