# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

'''
Model bank - deep convolutional neural network architectures
'''

from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten

from keras.models import Model

import os
import datetime
import h5py

def mnist(n_features, dim1, dim2, n_channels=1, weights_path = None):

    x = Input(shape=(dim1, dim2, n_channels))
    
    layer1 = Convolution2D(32, kernel_size=(3, 3), activation="relu")(x)
    layer1 = MaxPooling2D(pool_size=(2, 2))(layer1)
    layer2 = Convolution2D(64, kernel_size=(3, 3), activation="relu")(layer1)
    layer2 = MaxPooling2D(pool_size=(2, 2))(layer2)
    layer_flat = Flatten()(layer2)
    y = Dense(n_features, activation="softmax")(layer_flat)
        
    model = Model(x, y)
    
    if weights_path != None:
        model.load_weights(weights_path)
        
    return model