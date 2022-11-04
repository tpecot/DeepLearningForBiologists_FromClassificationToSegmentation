# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

'''
Model bank - deep convolutional neural network architectures
'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, concatenate, Dense, Activation, Convolution2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization, ZeroPadding2D, UpSampling2D, Conv2D, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop

from tensorflow.keras.models import Model

import os
import datetime
import h5py

def conv2d_bn(x, filters, num_row, num_col, border_mode='same', strides=(1, 1), data_format='channels_last', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Convolution2D(filters=filters, kernel_size=(num_row,num_col), strides=strides, padding=border_mode, data_format=data_format,name=conv_name)(x)
    if data_format=='channels_last':    
        x = BatchNormalization(axis=-1, name=bn_name)(x)
    else:
        x = BatchNormalization(axis=1, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

  

def get_core(dim1, dim2, nb_neurons_first_layer, n_channels):
    
    x = Input(shape=(dim1, dim2, n_channels))

    channel_axis = 3
    format = 'channels_last'

    down1 = conv2d_bn(x, nb_neurons_first_layer, 3, 3)
    down1 = conv2d_bn(down1, nb_neurons_first_layer, 3, 3)
    down1_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down1)

    down2 = conv2d_bn(down1_pool, 2*nb_neurons_first_layer, 3, 3)
    down2 = conv2d_bn(down2, 2*nb_neurons_first_layer, 3, 3)
    down2_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down2)

    down3 = conv2d_bn(down2_pool, 4*nb_neurons_first_layer, 3, 3)
    down3 = conv2d_bn(down3, 4*nb_neurons_first_layer, 3, 3)
    down3_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down3)

    center = conv2d_bn(down3_pool, 8*nb_neurons_first_layer, 3, 3)
    center = conv2d_bn(center, 8*nb_neurons_first_layer, 3, 3)

    up3 = UpSampling2D((2, 2))(center)
    up3 = concatenate([down3, up3], axis=channel_axis)
    up3 = conv2d_bn(up3, 4*nb_neurons_first_layer, 3, 3)
    up3 = conv2d_bn(up3, 4*nb_neurons_first_layer, 3, 3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=channel_axis)
    up2 = conv2d_bn(up2, 2*nb_neurons_first_layer, 3, 3)
    up2 = conv2d_bn(up2, 2*nb_neurons_first_layer, 3, 3)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=channel_axis)
    up1 = conv2d_bn(up1, nb_neurons_first_layer, 3, 3)
    up1 = conv2d_bn(up1, nb_neurons_first_layer, 3, 3)

    return [x, up1]


def get_core2(dim1, dim2, nb_neurons_first_layer, n_channels):
    
    x = Input(shape=(dim1, dim2, n_channels))

    channel_axis = 3
    format = 'channels_last'

    down1 = conv2d_bn(x, nb_neurons_first_layer, 3, 3)
    down1 = conv2d_bn(down1, nb_neurons_first_layer, 3, 3)
    down1_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down1)

    down2 = conv2d_bn(down1_pool, 2*nb_neurons_first_layer, 3, 3)
    down2 = conv2d_bn(down2, 2*nb_neurons_first_layer, 3, 3)
    down2_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down2)

    down3 = conv2d_bn(down2_pool, 4*nb_neurons_first_layer, 3, 3)
    down3 = conv2d_bn(down3, 4*nb_neurons_first_layer, 3, 3)
    down3_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down3)

    down4 = conv2d_bn(down3_pool, 8*nb_neurons_first_layer, 3, 3)
    down4 = conv2d_bn(down4, 8*nb_neurons_first_layer, 3, 3)
    down4_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down4)

    center = conv2d_bn(down4_pool, 16*nb_neurons_first_layer, 3, 3)
    center = conv2d_bn(center, 16*nb_neurons_first_layer, 3, 3)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=channel_axis)
    up4 = conv2d_bn(up4, 8*nb_neurons_first_layer, 3, 3)
    up4 = conv2d_bn(up4, 8*nb_neurons_first_layer, 3, 3)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=channel_axis)
    up3 = conv2d_bn(up3, 4*nb_neurons_first_layer, 3, 3)
    up3 = conv2d_bn(up3, 4*nb_neurons_first_layer, 3, 3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=channel_axis)
    up2 = conv2d_bn(up2, 2*nb_neurons_first_layer, 3, 3)
    up2 = conv2d_bn(up2, 2*nb_neurons_first_layer, 3, 3)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=channel_axis)
    up1 = conv2d_bn(up1, nb_neurons_first_layer, 3, 3)
    up1 = conv2d_bn(up1, nb_neurons_first_layer, 3, 3)

    return [x, up1]


def get_core3(dim1, dim2, nb_neurons_first_layer, n_channels):
    
    x = Input(shape=(dim1, dim2, n_channels))

    channel_axis = 3
    format = 'channels_last'

    down1 = conv2d_bn(x, nb_neurons_first_layer, 3, 3)
    down1 = conv2d_bn(down1, nb_neurons_first_layer, 3, 3)
    down1_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down1)

    down2 = conv2d_bn(down1_pool, 2*nb_neurons_first_layer, 3, 3)
    down2 = conv2d_bn(down2, 2*nb_neurons_first_layer, 3, 3)
    down2_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down2)

    down3 = conv2d_bn(down2_pool, 4*nb_neurons_first_layer, 3, 3)
    down3 = conv2d_bn(down3, 4*nb_neurons_first_layer, 3, 3)
    down3_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down3)

    down4 = conv2d_bn(down3_pool, 8*nb_neurons_first_layer, 3, 3)
    down4 = conv2d_bn(down4, 8*nb_neurons_first_layer, 3, 3)
    down4_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down4)

    down5 = conv2d_bn(down4_pool, 16*nb_neurons_first_layer, 3, 3)
    down5 = conv2d_bn(down5, 16*nb_neurons_first_layer, 3, 3)
    down5_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down5)

    center = conv2d_bn(down5_pool, 32*nb_neurons_first_layer, 3, 3)
    center = conv2d_bn(center, 32*nb_neurons_first_layer, 3, 3)

    up5 = UpSampling2D((2, 2))(center)
    up5 = concatenate([down5, up5], axis=channel_axis)
    up5 = conv2d_bn(up5, 16*nb_neurons_first_layer, 3, 3)
    up5 = conv2d_bn(up5, 16*nb_neurons_first_layer, 3, 3)

    up4 = UpSampling2D((2, 2))(up5)
    up4 = concatenate([down4, up4], axis=channel_axis)
    up4 = conv2d_bn(up4, 8*nb_neurons_first_layer, 3, 3)
    up4 = conv2d_bn(up4, 8*nb_neurons_first_layer, 3, 3)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=channel_axis)
    up3 = conv2d_bn(up3, 4*nb_neurons_first_layer, 3, 3)
    up3 = conv2d_bn(up3, 4*nb_neurons_first_layer, 3, 3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=channel_axis)
    up2 = conv2d_bn(up2, 2*nb_neurons_first_layer, 3, 3)
    up2 = conv2d_bn(up2, 2*nb_neurons_first_layer, 3, 3)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=channel_axis)
    up1 = conv2d_bn(up1, nb_neurons_first_layer, 3, 3)
    up1 = conv2d_bn(up1, nb_neurons_first_layer, 3, 3)

    return [x, up1]


def get_core4(dim1, dim2, nb_neurons_first_layer, n_channels):
    
    x = Input(shape=(dim1, dim2, n_channels))

    channel_axis = 3
    format = 'channels_last'

    down1 = conv2d_bn(x, nb_neurons_first_layer, 3, 3)
    down1 = conv2d_bn(down1, nb_neurons_first_layer, 3, 3)
    down1_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down1)

    down2 = conv2d_bn(down1_pool, 2*nb_neurons_first_layer, 3, 3)
    down2 = conv2d_bn(down2, 2*nb_neurons_first_layer, 3, 3)
    down2_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down2)

    down3 = conv2d_bn(down2_pool, 4*nb_neurons_first_layer, 3, 3)
    down3 = conv2d_bn(down3, 4*nb_neurons_first_layer, 3, 3)
    down3_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down3)

    down4 = conv2d_bn(down3_pool, 8*nb_neurons_first_layer, 3, 3)
    down4 = conv2d_bn(down4, 8*nb_neurons_first_layer, 3, 3)
    down4_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down4)

    down5 = conv2d_bn(down4_pool, 16*nb_neurons_first_layer, 3, 3)
    down5 = conv2d_bn(down5, 16*nb_neurons_first_layer, 3, 3)
    down5_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down5)

    down6 = conv2d_bn(down5_pool, 32*nb_neurons_first_layer, 3, 3)
    down6 = conv2d_bn(down6, 32*nb_neurons_first_layer, 3, 3)
    down6_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down6)

    center = conv2d_bn(down6_pool, 64*nb_neurons_first_layer, 3, 3)
    center = conv2d_bn(center, 64*nb_neurons_first_layer, 3, 3)

    up6 = UpSampling2D((2, 2))(center)
    up6 = concatenate([down6, up6], axis=channel_axis)
    up6 = conv2d_bn(up6, 32*nb_neurons_first_layer, 3, 3)
    up6 = conv2d_bn(up6, 32*nb_neurons_first_layer, 3, 3)

    up5 = UpSampling2D((2, 2))(up6)
    up5 = concatenate([down5, up5], axis=channel_axis)
    up5 = conv2d_bn(up5, 16*nb_neurons_first_layer, 3, 3)
    up5 = conv2d_bn(up5, 16*nb_neurons_first_layer, 3, 3)

    up4 = UpSampling2D((2, 2))(up5)
    up4 = concatenate([down4, up4], axis=channel_axis)
    up4 = conv2d_bn(up4, 8*nb_neurons_first_layer, 3, 3)
    up4 = conv2d_bn(up4, 8*nb_neurons_first_layer, 3, 3)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=channel_axis)
    up3 = conv2d_bn(up3, 4*nb_neurons_first_layer, 3, 3)
    up3 = conv2d_bn(up3, 4*nb_neurons_first_layer, 3, 3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=channel_axis)
    up2 = conv2d_bn(up2, 2*nb_neurons_first_layer, 3, 3)
    up2 = conv2d_bn(up2, 2*nb_neurons_first_layer, 3, 3)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=channel_axis)
    up1 = conv2d_bn(up1, nb_neurons_first_layer, 3, 3)
    up1 = conv2d_bn(up1, nb_neurons_first_layer, 3, 3)

    return [x, up1]

def get_core5(dim1, dim2, nb_neurons_first_layer, n_channels):
    
    x = Input(shape=(dim1, dim2, n_channels))

    channel_axis = 3
    format = 'channels_last'

    down1 = conv2d_bn(x, nb_neurons_first_layer, 3, 3)
    down1 = conv2d_bn(down1, nb_neurons_first_layer, 3, 3)
    down1_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down1)

    down2 = conv2d_bn(down1_pool, 2*nb_neurons_first_layer, 3, 3)
    down2 = conv2d_bn(down2, 2*nb_neurons_first_layer, 3, 3)
    down2_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down2)

    down3 = conv2d_bn(down2_pool, 4*nb_neurons_first_layer, 3, 3)
    down3 = conv2d_bn(down3, 4*nb_neurons_first_layer, 3, 3)
    down3_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down3)

    down4 = conv2d_bn(down3_pool, 8*nb_neurons_first_layer, 3, 3)
    down4 = conv2d_bn(down4, 8*nb_neurons_first_layer, 3, 3)
    down4_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down4)

    down5 = conv2d_bn(down4_pool, 16*nb_neurons_first_layer, 3, 3)
    down5 = conv2d_bn(down5, 16*nb_neurons_first_layer, 3, 3)
    down5_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down5)

    down6 = conv2d_bn(down5_pool, 32*nb_neurons_first_layer, 3, 3)
    down6 = conv2d_bn(down6, 32*nb_neurons_first_layer, 3, 3)
    down6_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down6)

    down7 = conv2d_bn(down6_pool, 64*nb_neurons_first_layer, 3, 3)
    down7 = conv2d_bn(down7, 64*nb_neurons_first_layer, 3, 3)
    down7_pool = MaxPooling2D(pool_size = (2, 2), data_format=format)(down7)

    center = conv2d_bn(down7_pool, 128*nb_neurons_first_layer, 3, 3)
    center = conv2d_bn(center, 128*nb_neurons_first_layer, 3, 3)

    up7 = UpSampling2D((2, 2))(center)
    up7 = concatenate([down7, up7], axis=channel_axis)
    up7 = conv2d_bn(up7, 64*nb_neurons_first_layer, 3, 3)
    up7 = conv2d_bn(up7, 64*nb_neurons_first_layer, 3, 3)

    up6 = UpSampling2D((2, 2))(up7)
    up6 = concatenate([down6, up6], axis=channel_axis)
    up6 = conv2d_bn(up6, 32*nb_neurons_first_layer, 3, 3)
    up6 = conv2d_bn(up6, 32*nb_neurons_first_layer, 3, 3)

    up5 = UpSampling2D((2, 2))(up6)
    up5 = concatenate([down5, up5], axis=channel_axis)
    up5 = conv2d_bn(up5, 16*nb_neurons_first_layer, 3, 3)
    up5 = conv2d_bn(up5, 16*nb_neurons_first_layer, 3, 3)

    up4 = UpSampling2D((2, 2))(up5)
    up4 = concatenate([down4, up4], axis=channel_axis)
    up4 = conv2d_bn(up4, 8*nb_neurons_first_layer, 3, 3)
    up4 = conv2d_bn(up4, 8*nb_neurons_first_layer, 3, 3)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=channel_axis)
    up3 = conv2d_bn(up3, 4*nb_neurons_first_layer, 3, 3)
    up3 = conv2d_bn(up3, 4*nb_neurons_first_layer, 3, 3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=channel_axis)
    up2 = conv2d_bn(up2, 2*nb_neurons_first_layer, 3, 3)
    up2 = conv2d_bn(up2, 2*nb_neurons_first_layer, 3, 3)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=channel_axis)
    up1 = conv2d_bn(up1, nb_neurons_first_layer, 3, 3)
    up1 = conv2d_bn(up1, nb_neurons_first_layer, 3, 3)

    return [x, up1]

def unet(n_features, dim1, dim2, n_channels=1, model_depth=3, nb_neurons_first_layer=64, weights_path = None):

    if int(model_depth)==3:
        [x, y] = get_core(dim1, dim2, nb_neurons_first_layer, n_channels)
    elif int(model_depth)==4:
        [x, y] = get_core2(dim1, dim2, nb_neurons_first_layer, n_channels)
    elif int(model_depth)==5:
        [x, y] = get_core3(dim1, dim2, nb_neurons_first_layer, n_channels)
    elif int(model_depth)==6:
        [x, y] = get_core4(dim1, dim2, nb_neurons_first_layer, n_channels)
    else:
        [x, y] = get_core5(dim1, dim2, nb_neurons_first_layer, n_channels)

    y = Convolution2D(filters=n_features, kernel_size=(1,1), activation = 'sigmoid', padding = 'same', data_format = 'channels_last')(y)

    model = Model(x, y)
    
    if weights_path != None:
        model.load_weights(weights_path)
        
    return model
