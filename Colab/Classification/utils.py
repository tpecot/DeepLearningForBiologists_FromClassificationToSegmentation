# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

"""
Functions needed to run the notebooks
"""

"""
Import python packages
"""

import numpy as np
import tensorflow as tf
import skimage

import sys
import os
from scipy import ndimage
import threading
from threading import Thread, Lock
import h5py
import csv

from skimage.io import imread, imsave
import skimage as sk
import tifffile as tiff
import cv2
  
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    
import datetime

import ipywidgets as widgets
import ipyfilechooser
from ipyfilechooser import FileChooser
from ipywidgets import HBox, Label, Layout

from models import mnist as mnist

"""
Interfaces
"""

def training_parameters_interface(nb_trainings):
    training_dir = np.zeros([nb_trainings], FileChooser)
    validation_dir = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    nb_channels = np.zeros([nb_trainings], HBox)
    nb_classes = np.zeros([nb_trainings], HBox)
    imaging_field_x = np.zeros([nb_trainings], HBox)
    imaging_field_y = np.zeros([nb_trainings], HBox)
    learning_rate = np.zeros([nb_trainings], HBox)
    nb_epochs = np.zeros([nb_trainings], HBox)
    nb_augmentations = np.zeros([nb_trainings], HBox)
    batch_size = np.zeros([nb_trainings], HBox)
    train_to_val_ratio = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Training directory")
        training_dir[i] = FileChooser('./datasets')
        display(training_dir[i])
        print('\x1b[1m'+"Validation directory")
        validation_dir[i] = FileChooser('./datasets')
        display(validation_dir[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./models')
        display(output_dir[i])

        label_layout = Layout(width='180px',height='30px')

        nb_channels[i] = HBox([Label('Number of channels:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_channels[i])

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        imaging_field_x[i] = HBox([Label('Imaging field in x:', layout=label_layout), widgets.IntText(
            value=256, description='', disabled=False)])
        display(imaging_field_x[i])

        imaging_field_y[i] = HBox([Label('Imaging field in y:', layout=label_layout), widgets.IntText(
            value=256, description='', disabled=False)])
        display(imaging_field_y[i])

        learning_rate[i] = HBox([Label('Learning rate:', layout=label_layout), widgets.FloatText(
            value=1e-4, description='', disabled=False)])
        display(learning_rate[i])

        nb_epochs[i] = HBox([Label('Number of epochs:', layout=label_layout), widgets.IntText(
            value=100, description='', disabled=False)])
        display(nb_epochs[i])

        nb_augmentations[i] = HBox([Label('Number of augmentations:', layout=label_layout), widgets.IntText(
            value=0, description='', disabled=False)])
        display(nb_augmentations[i])

        batch_size[i] = HBox([Label('Batch size:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(batch_size[i])

        train_to_val_ratio[i] = HBox([Label('Ratio of training in validation:', layout=label_layout), widgets.BoundedFloatText(
            value=0.2, min=0.01, max=0.99, step=0.01, description='', disabled=False, color='black'
        )])
        display(train_to_val_ratio[i])

    parameters.append(training_dir)
    parameters.append(validation_dir)
    parameters.append(output_dir)
    parameters.append(nb_channels)
    parameters.append(nb_classes)
    parameters.append(imaging_field_x)
    parameters.append(imaging_field_y)
    parameters.append(learning_rate)
    parameters.append(nb_epochs)
    parameters.append(nb_augmentations)
    parameters.append(batch_size)
    parameters.append(train_to_val_ratio)
    
    return parameters  

def training_parameters_TL_interface(nb_trainings):
    training_dir = np.zeros([nb_trainings], FileChooser)
    validation_dir = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    nb_channels = np.zeros([nb_trainings], HBox)
    nb_classes = np.zeros([nb_trainings], HBox)
    imaging_field_x = np.zeros([nb_trainings], HBox)
    imaging_field_y = np.zeros([nb_trainings], HBox)
    transfer_learning = np.zeros([nb_trainings], HBox)
    last_layer_training = np.zeros([nb_trainings], HBox)
    nb_epochs_last_layer = np.zeros([nb_trainings], HBox)
    learning_rate_last_layer = np.zeros([nb_trainings], HBox)
    last_block_training = np.zeros([nb_trainings], HBox)
    nb_epochs_last_block = np.zeros([nb_trainings], HBox)
    learning_rate_last_block = np.zeros([nb_trainings], HBox)
    all_network_training = np.zeros([nb_trainings], HBox)
    nb_epochs_all = np.zeros([nb_trainings], HBox)
    learning_rate_all = np.zeros([nb_trainings], HBox)
    nb_augmentations = np.zeros([nb_trainings], HBox)
    batch_size = np.zeros([nb_trainings], HBox)
    train_to_val_ratio = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Training directory")
        training_dir[i] = FileChooser('./datasets')
        display(training_dir[i])
        print('\x1b[1m'+"Validation directory")
        validation_dir[i] = FileChooser('./datasets')
        display(validation_dir[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./models')
        display(output_dir[i])

        label_layout = Layout(width='250px',height='30px')

        nb_channels[i] = HBox([Label('Number of channels:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_channels[i])

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        imaging_field_x[i] = HBox([Label('Imaging field in x:', layout=label_layout), widgets.IntText(
            value=256, description='', disabled=False)])
        display(imaging_field_x[i])

        imaging_field_y[i] = HBox([Label('Imaging field in y:', layout=label_layout), widgets.IntText(
            value=256, description='', disabled=False)])
        display(imaging_field_y[i])

        transfer_learning[i] = HBox([Label('ImageNet transfer learning:', layout=label_layout), widgets.Checkbox(
            value=True, description='',disabled=False)])
        display(transfer_learning[i])

        last_layer_training[i] = HBox([Label('Last layer training:', layout=label_layout), widgets.Checkbox(
            value=True, description='',disabled=False)])
        display(last_layer_training[i])

        nb_epochs_last_layer[i] = HBox([Label('Number of epochs for last layer training:', layout=label_layout), widgets.IntText(
            value=20, description='', disabled=False)])
        display(nb_epochs_last_layer[i])

        learning_rate_last_layer[i] = HBox([Label('Learning rate for last layer training:', layout=label_layout), widgets.FloatText(
            value=0.001, description='', disabled=False)])
        display(learning_rate_last_layer[i])

        last_block_training[i] = HBox([Label('Last block training:', layout=label_layout), widgets.Checkbox(
            value=False, description='',disabled=False)])
        display(last_block_training[i])

        nb_epochs_last_block[i] = HBox([Label('Number of epochs for last block training:', layout=label_layout), widgets.IntText(value=50, description='', disabled=False)])
        display(nb_epochs_last_block[i])

        learning_rate_last_block[i] = HBox([Label('Learning rate for last block training:', layout=label_layout), widgets.FloatText(
            value=0.0005, description='', disabled=False)])
        display(learning_rate_last_block[i])

        all_network_training[i] = HBox([Label('Training all network:', layout=label_layout), widgets.Checkbox(
            value=False, description='',disabled=False)])
        display(all_network_training[i])

        nb_epochs_all[i] = HBox([Label('Number of epochs for all network training:', layout=label_layout), widgets.IntText(
            value=100, description='', disabled=False)])
        display(nb_epochs_all[i])

        learning_rate_all[i] = HBox([Label('Learning rate for all network training:', layout=label_layout), widgets.FloatText(
            value=0.0001, description='', disabled=False)])
        display(learning_rate_all[i])

        nb_augmentations[i] = HBox([Label('Number of augmentations:', layout=label_layout), widgets.IntText(
            value=0, description='', disabled=False)])
        display(nb_augmentations[i])

        batch_size[i] = HBox([Label('Batch size:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(batch_size[i])

        train_to_val_ratio[i] = HBox([Label('Ratio of training in validation:', layout=label_layout), widgets.BoundedFloatText(
            value=0.2, min=0.01, max=0.99, step=0.01, description='', disabled=False, color='black'
        )])
        display(train_to_val_ratio[i])

    parameters.append(training_dir)
    parameters.append(validation_dir)
    parameters.append(output_dir)
    parameters.append(nb_channels)
    parameters.append(nb_classes)
    parameters.append(imaging_field_x)
    parameters.append(imaging_field_y)
    parameters.append(transfer_learning)
    parameters.append(last_layer_training)
    parameters.append(nb_epochs_last_layer)
    parameters.append(learning_rate_last_layer)
    parameters.append(last_block_training)
    parameters.append(nb_epochs_last_block)
    parameters.append(learning_rate_last_block)
    parameters.append(all_network_training)
    parameters.append(nb_epochs_all)
    parameters.append(learning_rate_all)
    parameters.append(nb_augmentations)
    parameters.append(batch_size)
    parameters.append(train_to_val_ratio)
    
    return parameters  

def running_parameters_interface(nb_trainings):
    input_dir = np.zeros([nb_trainings], FileChooser)
    input_classifier = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    output_mode = np.zeros([nb_trainings], HBox)
    nb_channels = np.zeros([nb_trainings], HBox)
    nb_classes = np.zeros([nb_trainings], HBox)
    imaging_field_x = np.zeros([nb_trainings], HBox)
    imaging_field_y = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Input directory")
        input_dir[i] = FileChooser('./datasets')
        display(input_dir[i])
        print('\x1b[1m'+"Input classifier")
        input_classifier[i] = FileChooser('./models')
        display(input_classifier[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./datasets')
        display(output_dir[i])

        label_layout = Layout(width='150px',height='30px')

        output_mode[i] = HBox([Label('Score:', layout=label_layout), widgets.Checkbox(
            value=False, description='',disabled=False)])
        display(output_mode[i])
        
        nb_channels[i] = HBox([Label('Number of channels:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_channels[i])

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        imaging_field_x[i] = HBox([Label('Imaging field in x:', layout=label_layout), widgets.IntText(
            value=256, description='', disabled=False)])
        display(imaging_field_x[i])

        imaging_field_y[i] = HBox([Label('Imaging field in y:', layout=label_layout), widgets.IntText(
            value=256, description='', disabled=False)])
        display(imaging_field_y[i])

    parameters.append(input_dir)
    parameters.append(input_classifier)
    parameters.append(output_dir)
    parameters.append(output_mode)
    parameters.append(nb_channels)
    parameters.append(nb_classes)
    parameters.append(imaging_field_x)
    parameters.append(imaging_field_y)
    
    return parameters  

       
"""
Training and processing calling functions 
""" 
def training_MobileNetV2(nb_trainings, parameters):
    for i in range(nb_trainings):
        if parameters[0][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an input directory for training")
        if parameters[2][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an output directory for the trained classifier")
    
        pretrained = True
        last_layers_first = True
        last_part_network = True
        whole_network = False
        
        if pretrained:
            # pre-trained
            # create the base pre-trained model
            base_model = MobileNetV2(weights='imagenet', include_top=False)

            # add a global spatial average pooling layer
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            # let's add a fully-connected layer
            x = Dense(1024, activation='relu')(x)
            # and a logistic layer -- let's say we have 200 classes
            predictions = Dense(parameters[4][i].children[1].value, activation='softmax')(x)

            # this is the model we will train
            model = Model(inputs=base_model.input, outputs=predictions)

            if parameters[8][i].children[1].value:
                if parameters[11][i].children[1].value:
                    if parameters[14][i].children[1].value:
                        model_name = "MobileNetV2_withTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[10][i].children[1].value)+"_"+str(parameters[9][i].children[1].value)+"ep_last_layer"+"_lr_"+str(parameters[13][i].children[1].value)+"_"+str(parameters[12][i].children[1].value)+"ep_last_block"+"_lr_"+str(parameters[16][i].children[1].value)+"_"+str(parameters[15][i].children[1].value)+"ep_whole_network_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    else:
                        model_name = "MobileNetV2_withTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[10][i].children[1].value)+"_"+str(parameters[9][i].children[1].value)+"ep_last_layer_lr_"+str(parameters[13][i].children[1].value)+"_"+str(parameters[12][i].children[1].value)+"ep_last_block_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                else:
                    if parameters[14][i].children[1].value:
                        model_name = "MobileNetV2_withTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[10][i].children[1].value)+"_"+str(parameters[9][i].children[1].value)+"ep_last_layer"+"_lr_"+str(parameters[16][i].children[1].value)+"_"+str(parameters[15][i].children[1].value)+"ep_whole_network_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    else:
                        model_name = "MobileNetV2_withTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[10][i].children[1].value)+"_"+str(parameters[9][i].children[1].value)+"ep_last_layer_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            else:
                if parameters[11][i].children[1].value:
                    if parameters[14][i].children[1].value:
                        model_name = "MobileNetV2_withTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[13][i].children[1].value)+"_"+str(parameters[12][i].children[1].value)+"ep_last_block"+"_lr_"+str(parameters[16][i].children[1].value)+"_"+str(parameters[15][i].children[1].value)+"ep_whole_network_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    else:
                        model_name = "MobileNetV2_withTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[13][i].children[1].value)+"_"+str(parameters[12][i].children[1].value)+"ep_last_block_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                else:
                    if parameters[14][i].children[1].value:
                        model_name = "MobileNetV2_withTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[16][i].children[1].value)+"_"+str(parameters[15][i].children[1].value)+"ep_whole_network_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    else:
                        sys.exit("Training #"+str(i+1)+": You need to select a part of the network to train")
                        
            if parameters[8][i].children[1].value:
                
                # first: train only the top layers (which were randomly initialized)
                # i.e. freeze all convolutional layers
                for layer in base_model.layers:
                    layer.trainable = False
                
                train_model_sample(model, parameters[0][i].selected, parameters[1][i].selected, model_name, parameters[18][i].children[1].value, parameters[9][i].children[1].value, parameters[5][i].children[1].value, parameters[6][i].children[1].value, parameters[3][i].children[1].value, parameters[2][i].selected, parameters[10][i].children[1].value, parameters[17][i].children[1].value, parameters[19][i].children[1].value)

            if parameters[11][i].children[1].value:
                for layer in model.layers[:143]:
                    layer.trainable = False
                for layer in model.layers[143:]:
                    layer.trainable = True

                train_model_sample(model, parameters[0][i].selected, parameters[1][i].selected, model_name, parameters[18][i].children[1].value, parameters[12][i].children[1].value, parameters[5][i].children[1].value, parameters[6][i].children[1].value, parameters[3][i].children[1].value, parameters[2][i].selected, parameters[13][i].children[1].value, parameters[17][i].children[1].value, parameters[19][i].children[1].value)
                
            if parameters[14][i].children[1].value:
                # second: train all layers
                for layer in base_model.layers:
                    layer.trainable = True

                train_model_sample(model, parameters[0][i].selected, parameters[1][i].selected, model_name, parameters[18][i].children[1].value, parameters[15][i].children[1].value, parameters[5][i].children[1].value, parameters[6][i].children[1].value, parameters[3][i].children[1].value, parameters[2][i].selected, parameters[16][i].children[1].value, parameters[17][i].children[1].value, parameters[19][i].children[1].value)
                
        else:
            base_model = MobileNetV2(include_top=False)

            # add a global spatial average pooling layer
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            # let's add a fully-connected layer
            x = Dense(1024, activation='relu')(x)
            # and a logistic layer -- let's say we have 200 classes
            predictions = Dense(parameters[4][i].children[1].value, activation='softmax')(x)

            # this is the model we will train
            model = Model(inputs=base_model.input, outputs=predictions)

            if parameters[8][i].children[1].value:
                if parameters[11][i].children[1].value:
                    if parameters[14][i].children[1].value:
                        model_name = "MobileNetV2_withoutTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[10][i].children[1].value)+"_"+str(parameters[9][i].children[1].value)+"ep_last_layer"+"_lr_"+str(parameters[13][i].children[1].value)+"_"+str(parameters[12][i].children[1].value)+"ep_last_block"+"_lr_"+str(parameters[16][i].children[1].value)+"_"+str(parameters[15][i].children[1].value)+"ep_whole_network_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    else:
                        model_name = "MobileNetV2_withoutTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[10][i].children[1].value)+"_"+str(parameters[9][i].children[1].value)+"ep_last_layer"+"_lr_"+str(parameters[13][i].children[1].value)+"_"+str(parameters[12][i].children[1].value)+"ep_last_block"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                else:
                    if parameters[14][i].children[1].value:
                        model_name = "MobileNetV2_withoutTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[10][i].children[1].value)+"_"+str(parameters[9][i].children[1].value)+"ep_last_layer"+"_lr_"+str(parameters[16][i].children[1].value)+"_"+str(parameters[15][i].children[1].value)+"ep_whole_network_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    else:
                        model_name = "MobileNetV2_withoutTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[10][i].children[1].value)+"_"+str(parameters[9][i].children[1].value)+"ep_last_layer"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            else:
                if parameters[11][i].children[1].value:
                    if parameters[14][i].children[1].value:
                        model_name = "MobileNetV2_withoutTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[13][i].children[1].value)+"_"+str(parameters[12][i].children[1].value)+"ep_last_block"+"_lr_"+str(parameters[16][i].children[1].value)+"_"+str(parameters[15][i].children[1].value)+"ep_whole_network_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    else:
                        model_name = "MobileNetV2_withoutTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[13][i].children[1].value)+"_"+str(parameters[12][i].children[1].value)+"ep_last_block"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                else:
                    if parameters[14][i].children[1].value:
                        model_name = "MobileNetV2_withoutTL_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[16][i].children[1].value)+"_"+str(parameters[15][i].children[1].value)+"ep_whole_network_"+str(parameters[17][i].children[1].value)+"DA_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    else:
                        sys.exit("Training #"+str(i+1)+": You need to select a part of the network to train")
            
            if parameters[8][i].children[1].value:
                
                # first: train only the top layers (which were randomly initialized)
                # i.e. freeze all convolutional layers
                for layer in base_model.layers:
                    layer.trainable = False

                train_model_sample(model, parameters[0][i].selected, parameters[1][i].selected, model_name, parameters[18][i].children[1].value, parameters[9][i].children[1].value, parameters[5][i].children[1].value, parameters[6][i].children[1].value, parameters[3][i].children[1].value, parameters[2][i].selected, parameters[10][i].children[1].value, parameters[17][i].children[1].value, parameters[19][i].children[1].value)

            if parameters[11][i].children[1].value:
                for layer in model.layers[:143]:
                    layer.trainable = False
                for layer in model.layers[143:]:
                    layer.trainable = True

                train_model_sample(model, parameters[0][i].selected, parameters[1][i].selected, model_name, parameters[18][i].children[1].value, parameters[12][i].children[1].value, parameters[5][i].children[1].value, parameters[6][i].children[1].value, parameters[3][i].children[1].value, parameters[2][i].selected, parameters[13][i].children[1].value, parameters[17][i].children[1].value, parameters[19][i].children[1].value)
                
            if parameters[14][i].children[1].value:
                # second: train all layers
                for layer in base_model.layers:
                    layer.trainable = True

                train_model_sample(model, parameters[0][i].selected, parameters[1][i].selected, model_name, parameters[18][i].children[1].value, parameters[15][i].children[1].value, parameters[5][i].children[1].value, parameters[6][i].children[1].value, parameters[3][i].children[1].value, parameters[2][i].selected, parameters[16][i].children[1].value, parameters[17][i].children[1].value, parameters[19][i].children[1].value)
        
        del model

def training_mnist(nb_trainings, parameters):
    for i in range(nb_trainings):
        if parameters[0][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an input directory for training")
        if parameters[2][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an output directory for the trained classifier")
    
        model = mnist(parameters[4][i].children[1].value, parameters[5][i].children[1].value, parameters[6][i].children[1].value, parameters[3][i].children[1].value)
        
        model_name = "Mnist_"+str(parameters[3][i].children[1].value)+"ch_"+str(parameters[4][i].children[1].value)+"cl_"+str(parameters[5][i].children[1].value)+"_"+str(parameters[6][i].children[1].value)+"_lr_"+str(parameters[7][i].children[1].value)+"_"+str(parameters[9][i].children[1].value)+"DA_"+str(parameters[8][i].children[1].value)+"ep_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        train_model_sample(model, parameters[0][i].selected, parameters[1][i].selected, model_name,parameters[10][i].children[1].value, parameters[8][i].children[1].value, parameters[5][i].children[1].value, parameters[6][i].children[1].value, parameters[3][i].children[1].value, parameters[2][i].selected, parameters[7][i].children[1].value, parameters[9][i].children[1].value, parameters[11][i].children[1].value)
        del model
        

def running_mnist(nb_runnings, parameters):
    for i in range(nb_runnings):
        if parameters[0][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an input directory for images to be processed")
        if parameters[1][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select a trained classifier to run your images")
        if parameters[2][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an output directory for processed images")

        model = mnist(parameters[5][i].children[1].value, parameters[6][i].children[1].value, parameters[7][i].children[1].value, parameters[4][i].children[1].value, parameters[1][i].selected)
        run_models_on_directory(parameters[0][i].selected, parameters[2][i].selected, model, parameters[3][i].children[1].value, parameters[6][i].children[1].value, parameters[7][i].children[1].value, parameters[4][i].children[1].value, parameters[5][i].children[1].value)
        del model

def running_MobileNetV2(nb_runnings, parameters):
    for i in range(nb_runnings):
        if parameters[0][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an input directory for images to be processed")
        if parameters[1][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select a trained classifier to run your images")
        if parameters[2][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an output directory for processed images")

        base_model = MobileNetV2(weights='imagenet', include_top=False)
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(parameters[5][i].children[1].value, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(parameters[1][i].selected)
        run_models_on_directory(parameters[0][i].selected, parameters[2][i].selected, model, parameters[3][i].children[1].value, parameters[6][i].children[1].value, parameters[7][i].children[1].value, parameters[4][i].children[1].value, parameters[5][i].children[1].value)
        del model

        
        
"""
Useful functions 
"""
def search_label_for_image_in_file(file_name, image_to_search, nb_classes):
    labels = np.zeros((nb_classes), 'int32')
    found_image = False
    # Open the file in read only mode
    with open(file_name, newline='', encoding='utf-8', errors='ignore') as csvfile:
        read_obj = csv.reader(csvfile)
        for line in read_obj:
            if line[0]==image_to_search:
                found_image = True
                if int(line[1]) == -1:
                    labels[0] = 1
                else:
                    labels[int(line[1])] = 1
                #label_sum = 0
                #for i in range(1, nb_classes):
                #    labels[i-1] = int(line[i])
                #    label_sum += int(line[i])
                #if label_sum==0:
                #    labels[nb_classes-1] = 1
    #if found_image==False:
    #    print("Didn't find it")
    #    labels[nb_classes-1] = 1
    #found_image = True
    return found_image, labels

def rate_scheduler(lr = .001, decay = 0.95):
    def output_fn(epoch):
        epoch = np.int(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn

def process_image(img):
    
    if img.shape[2] == 1:
        output = np.zeros((img.shape[0], img.shape[1], 1), 'float32')
        
        percentile = 98.
        high = np.percentile(img, percentile)
        low = np.percentile(img, 100-percentile)

        output = np.minimum(high, img)
        output = np.maximum(low, output)

        output = (output - low) / (high - low)

    else:
        output = np.zeros((img.shape[0], img.shape[1], img.shape[2]), 'float32')
        for i in range(img.shape[2]):
            percentile = 98.
            high = np.percentile(img[:,:,i], percentile)
            low = np.percentile(img[:,:,i], 100-percentile)
            
            output[:,:,i] = np.minimum(high, img[:,:,i])
            output[:,:,i] = np.maximum(low, output[:,:,i])
            
            if high>low:
                output[:,:,i] = (output[:,:,i] - low) / (high - low)

    return output

def getfiles(direc_name):
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if ('.png'  in i ) or ('.jpg'  in i ) or ('.tif' in i) or ('tiff' in i)]

    imgfiles = imgfiles
    return imgfiles

def get_image(file_name):
    if ('.tif' in file_name) or ('tiff' in file_name):
        im = tiff.imread(file_name)
        #im = bytescale(im)
        im = np.float32(im)
    else:
        im = cv2.imread(file_name) 
        #im = np.float32(imread(file_name))
        
    if len(im.shape) < 3:
        output_im = np.zeros((im.shape[0], im.shape[1], 1))
        output_im[:, :, 0] = im
        im = output_im
    else:
        if im.shape[0]<im.shape[2]:
            output_im = np.zeros((im.shape[1], im.shape[2], im.shape[0]))
            for i in range(im.shape[0]):
                output_im[:, :, i] = im[i, :, :]
            im = output_im
    
    return im

"""
Data generator for training_data
"""

def get_data_sample(training_directory, validation_directory, nb_channels = 1, nb_classes = 3, imaging_field_x = 256, imaging_field_y = 256, nb_augmentations = 1, validation_training_ratio = 0.1):

    channels_training = []
    labels_training = []
    channels_validation = []
    labels_validation = []
    final_channels_training = []
    final_labels_training = []

    imglist_training_directory = os.path.join(training_directory)

    # adding evaluation data into validation
    if validation_directory is not None:

        imglist_validation_directory = os.path.join(validation_directory)

        imageValFileList = [f for f in os.listdir(imglist_validation_directory) if ('.png'  in f ) or ('.jpg'  in f ) or ('.tif' in f) or ('tiff' in f)]
        
        for imageFile in imageValFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]

            found_image, current_labels = search_label_for_image_in_file(os.path.join(imglist_validation_directory, "labels.csv"), imageFile, nb_classes)
            if found_image==False:
                sys.exit("The image " + baseName + " does not have labels in labels.csv")
            labels_validation.append(current_labels)
            
            imagePath = os.path.join(imglist_validation_directory, imageFile)
            current_image = get_image(imagePath)
            
            if current_image.shape[0]<imaging_field_x:
                sys.exit("The image " + baseName + " has a smaller x dimension than the imaging field")
            if current_image.shape[1]<imaging_field_y:
                sys.exit("The image " + baseName + " has a smaller y dimension than the imaging field")
            if current_image.shape[2]!=nb_channels:
                sys.exit("The image " + baseName + " has a different number of channels than indicated in the U-Net architecture")
            
            channels_validation.append(process_image(current_image))

        imageFileList = [f for f in os.listdir(imglist_training_directory) if ('.png'  in f ) or ('.jpg'  in f ) or ('.tif' in f) or ('tiff' in f)]
        
        for imageFile in imageFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            
            found_image, current_labels = search_label_for_image_in_file(os.path.join(imglist_training_directory, "labels.csv"), imageFile, nb_classes)
            if found_image==False:
                sys.exit("The image " + baseName + " does not have labels in labels.csv")
            labels_training.append(current_labels)

            imagePath = os.path.join(imglist_training_directory, imageFile)
            current_image = get_image(imagePath)
            
            if current_image.shape[0]<imaging_field_x:
                sys.exit("The image " + baseName + " has a smaller x dimension than the imaging field")
            if current_image.shape[1]<imaging_field_y:
                sys.exit("The image " + baseName + " has a smaller y dimension than the imaging field")
            if current_image.shape[2]!=nb_channels:
                sys.exit("The image " + baseName + " has a different number of channels than indicated in the U-Net architecture")
            
            channels_training.append(process_image(current_image))
        
        # balance classes
        classes_for_training = np.zeros((len(labels_training)), dtype = 'uint16')
        nb_training_instances_per_class = np.zeros((nb_classes), dtype = 'uint16')
        for k in range(len(labels_training)):
            for i in range(nb_classes):
                if labels_training[k][i]>0:
                    nb_training_instances_per_class[i] += 1
                    classes_for_training[k] = i
        min_number_instances_per_class = np.min(nb_training_instances_per_class)
        for i in range(nb_classes):
            current_indices = np.where(classes_for_training==i)
            if nb_training_instances_per_class[i]==min_number_instances_per_class:
                for j in range(len(current_indices[0])):
                    final_channels_training.append(channels_training[int(list(current_indices[0])[j])])
                    final_labels_training.append(labels_training[int(list(current_indices[0])[j])])
            else:
                subset_indices = random.sample(list(current_indices[0]), min_number_instances_per_class)
                for j in range(min_number_instances_per_class):
                    final_channels_training.append(channels_training[subset_indices[j]])
                    final_labels_training.append(labels_training[subset_indices[j]])
                    
                    
    else:
        imageValFileList = [f for f in os.listdir(imglist_training_directory) if ('.png'  in f ) or ('.jpg'  in f ) or ('.tif' in f) or ('tiff' in f)]
        for imageFile in imageValFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]

            found_image, current_labels = search_label_for_image_in_file(os.path.join(imglist_training_directory, "labels.csv"), imageFile, nb_classes)
            if found_image==False:
                sys.exit("The image " + baseName + " does not have labels in labels.csv")
            labels_training.append(current_labels)
            
            imagePath = os.path.join(imglist_training_directory, imageFile)
            current_image = get_image(imagePath)
            
            if current_image.shape[0]<imaging_field_x:
                sys.exit("The image " + baseName + " has a smaller x dimension than the imaging field")
            if current_image.shape[1]<imaging_field_y:
                sys.exit("The image " + baseName + " has a smaller y dimension than the imaging field")
            if current_image.shape[2]!=nb_channels:
                sys.exit("The image " + baseName + " has a different number of channels than indicated in the U-Net architecture")
        
            channels_training.append(process_image(current_image))
        
        # balance classes
        classes_for_training = np.zeros((len(labels_training)), dtype = 'uint16')
        nb_training_instances_per_class = np.zeros((nb_classes), dtype = 'uint16')
        for k in range(len(labels_training)):
            for i in range(nb_classes):
                if labels_training[k][i]>0:
                    nb_training_instances_per_class[i] += 1
                    classes_for_training[k] = i
        min_number_instances_per_class = np.min(nb_training_instances_per_class)
        for i in range(nb_classes):
            current_indices = np.where(classes_for_training==i)
            if nb_training_instances_per_class[i]==min_number_instances_per_class:
                for j in range(len(current_indices[0])):
                    final_channels_training.append(channels_training[int(list(current_indices[0])[j])])
                    final_labels_training.append(labels_training[int(list(current_indices[0])[j])])
            else:
                subset_indices = random.sample(list(current_indices[0]), min_number_instances_per_class)
                for j in range(min_number_instances_per_class):
                    final_channels_training.append(channels_training[subset_indices[j]])
                    final_labels_training.append(labels_training[subset_indices[j]])

        channels_training = []
        labels_training = []
        for i in range(len(final_channels_training)):
            if random.Random().random() > validation_training_ratio:
                channels_training.append(final_channels_training[i])
                labels_training.append(final_labels_training[i])
            else:
                channels_validation.append(final_channels_training[i])
                labels_validation.append(final_labels_training[i])
        final_channels_training = channels_training
        final_labels_training = labels_training

                
    if len(final_channels_training) < 1:
        sys.exit("Empty train image list")

    #just to be non-empty
    #if len(channels_validation) < 1:
    #    channels_validation += channels_training[len(channels_training)-1]
    #    labels_validation += channels_validation[len(channels_validation)-1]
    

    X_test = channels_validation
    Y_test =[]
    for k in range(len(X_test)):
        X_test[k] = X_test[k][0:imaging_field_x, 0:imaging_field_y, :]
        Y_test.append(labels_validation[k])
        
    #train_dict = {"channels": channels_training, "labels": labels_training}
    train_dict = {"channels": final_channels_training, "labels": final_labels_training}

    return train_dict, (np.asarray(X_test).astype('float32'), np.asarray(Y_test).astype('float32'))




def random_sample_generator(x_init, y_init, batch_size, n_channels, n_classes, dim1, dim2):

    cpt = 0

    n_images = len(x_init)
    arr = np.arange(n_images)
    np.random.shuffle(arr)
    
    while(True):

        # buffers for a batch of data
        x = np.zeros((batch_size, dim1, dim2, n_channels))
        y = np.zeros((batch_size, n_classes))
        
        for k in range(batch_size):

            # get random image
            img_index = arr[cpt%n_images]

            # open image
            x_big = x_init[img_index]

            # get random crop
            if dim1==x_big.shape[0]:
                start_dim1 = 0
            else:
                start_dim1 = np.random.randint(low=0, high=x_big.shape[0] - dim1)
            if dim2==x_big.shape[1]:
                start_dim2 = 0
            else:
                start_dim2 = np.random.randint(low=0, high=x_big.shape[1] - dim2)

            patch_x = x_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
            patch_x = np.asarray(patch_x).astype('float32')
            
            # define label associated with image
            current_classes = np.asarray(y_init[img_index]).astype('float32')

            # save image to buffer
            x[k, :, :, :] = patch_x
            y[k, :] = current_classes
            cpt += 1

        # return the buffer
        yield(x, y)

        
def GenerateRandomImgaugAugmentation(
        pAugmentationLevel=5,           # number of augmentations
        pEnableFlipping1=True,          # enable x flipping
        pEnableFlipping2=True,          # enable y flipping
        pEnableRotation90=True,           # enable rotation
        pEnableRotation=False,           # enable rotation
        pMaxRotationDegree=15,             # maximum rotation degree
        pEnableShearX=False,             # enable x shear
        pEnableShearY=False,             # enable y shear
        pMaxShearDegree=15,             # maximum shear degree
        pEnableBlur=True,               # enable gaussian blur
        pBlurSigma=.5,                  # maximum sigma for gaussian blur
        pEnableDropOut=True,
        pMaxDropoutPercentage=0.01,
        pEnableSharpness=False,          # enable sharpness
        pSharpnessFactor=0.0001,           # maximum additional sharpness
        pEnableEmboss=False,             # enable emboss
        pEmbossFactor=0.0001,              # maximum emboss
        pEnableBrightness=False,         # enable brightness
        pBrightnessFactor=0.000001,         # maximum +- brightness
        pEnableRandomNoise=True,        # enable random noise
        pMaxRandomNoise=0.01,           # maximum random noise strength
        pEnableInvert=False,             # enables color invert
        pEnableContrast=True,           # enable contrast change
        pContrastFactor=0.01,            # maximum +- contrast
):
    
    augmentationMap = []
    augmentationMapOutput = []


    if pEnableFlipping1:
        aug = iaa.Fliplr()
        augmentationMap.append(aug)
        
    if pEnableFlipping2:
        aug = iaa.Flipud()
        augmentationMap.append(aug)

    if pEnableRotation90:
        randomNumber = random.Random().randint(1,3)
        aug = iaa.Rot90(randomNumber)
        augmentationMap.append(aug)

    if pEnableRotation:
        if random.Random().randint(0, 1)==1:
            randomRotation = random.Random().random()*pMaxRotationDegree
        else:
            randomRotation = -random.Random().random()*pMaxRotationDegree
        aug = iaa.Rotate(randomRotation)
        augmentationMap.append(aug)

    if pEnableShearX:
        if random.Random().randint(0, 1)==1:
            randomShearingX = random.Random().random()*pMaxShearDegree
        else:
            randomShearingX = -random.Random().random()*pMaxShearDegree
        aug = iaa.ShearX(randomShearingX)
        augmentationMap.append(aug)

    if pEnableShearY:
        if random.Random().randint(0, 1)==1:
            randomShearingY = random.Random().random()*pMaxShearDegree
        else:
            randomShearingY = -random.Random().random()*pMaxShearDegree
        aug = iaa.ShearY(randomShearingY)
        augmentationMap.append(aug)

    if pEnableDropOut:
        randomDropOut = random.Random().random()*pMaxDropoutPercentage
        aug = iaa.Dropout(p=randomDropOut, per_channel=False)
        augmentationMap.append(aug)

    if pEnableBlur:
        randomBlur = random.Random().random()*pBlurSigma
        aug = iaa.GaussianBlur(randomBlur)
        augmentationMap.append(aug)

    if pEnableSharpness:
        randomSharpness = random.Random().random()*pSharpnessFactor
        aug = iaa.Sharpen(randomSharpness)
        augmentationMap.append(aug)

    if pEnableEmboss:
        randomEmboss = random.Random().random()*pEmbossFactor
        aug = iaa.Emboss(randomEmboss)
        augmentationMap.append(aug)

    if pEnableBrightness:
        if random.Random().randint(0, 1)==1:
            randomBrightness = 1 - random.Random().random()*pBrightnessFactor
        else:
            randomBrightness = 1 + random.Random().random()*pBrightnessFactor
        aug = iaa.Add(randomBrightness)
        augmentationMap.append(aug)

    if pEnableRandomNoise:
        if random.Random().randint(0, 1)==1:
            randomNoise = 1 - random.Random().random()*pMaxRandomNoise
        else:
            randomNoise = 1 + random.Random().random()*pMaxRandomNoise
        aug = iaa.MultiplyElementwise(randomNoise,  per_channel=True)
        augmentationMap.append(aug)
        
    if pEnableInvert:
        aug = iaa.Invert(1)
        augmentationMap.append(aug)

    if pEnableContrast:
        if random.Random().randint(0, 1)==1:
            randomContrast = 1 - random.Random().random()*pContrastFactor
        else:
            randomContrast = 1 + random.Random().random()*pContrastFactor
        aug = iaa.contrast.LinearContrast(randomContrast)
        augmentationMap.append(aug)

    arr = np.arange(len(augmentationMap))
    np.random.shuffle(arr)
    for i in range(pAugmentationLevel):
        augmentationMapOutput.append(augmentationMap[arr[i]])
    
        
    return iaa.Sequential(augmentationMapOutput)

def random_sample_generator_dataAugmentation(x_init, y_init, batch_size, n_channels, n_classes, dim1, dim2, nb_augmentations):

    cpt = 0
    n_images = len(x_init)
    arr = np.arange(n_images)
    np.random.shuffle(arr)
    non_augmented_array = np.zeros(n_images)

    while(True):

        # buffers for a batch of data
        x = np.zeros((batch_size, dim1, dim2, n_channels))
        y = np.zeros((batch_size, n_classes))
        
        for k in range(batch_size):
            
            # get random image
            img_index = arr[cpt%n_images]

            # open images
            x_big = x_init[img_index]

            # augmentation
            augmentationMap = GenerateRandomImgaugAugmentation()

            x_aug = augmentationMap(image=x_big.astype('float32'))
            
            # image normalization
            x_norm = x_aug.astype('float32')

            # get random crop
            if dim1==x_big.shape[0]:
                start_dim1 = 0
            else:
                start_dim1 = np.random.randint(low=0, high=x_big.shape[0] - dim1)
            if dim2==x_big.shape[1]:
                start_dim2 = 0
            else:
                start_dim2 = np.random.randint(low=0, high=x_big.shape[1] - dim2)

            # non augmented image
            if non_augmented_array[cpt%n_images]==0:
                if random.Random().random() < (2./nb_augmentations):
                    non_augmented_array[cpt%n_images] = 1
                    x_aug = x_big
                    
            patch_x = x_aug[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
            patch_x = np.asarray(patch_x)

            # define label associated with image
            current_classes = np.asarray(y_init[img_index]).astype('float32')

            # save image to buffer
            x[k, :, :, :] = patch_x
            y[k, :] = current_classes


            cpt += 1
        

        # return the buffer
        yield(x, y)


"""
Training convnets
"""
def weighted_crossentropy(class_weights):

    def func(y_true, y_pred):
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        weighted_losses = weights * unweighted_losses
        return tf.reduce_mean(weighted_losses)

    return func


    
def train_model_sample(model = None, dataset_training = None,  dataset_validation = None,
                       model_name = "model", batch_size = 5, n_epoch = 100, 
                       imaging_field_x = 256, imaging_field_y = 256, n_channels = 1,
                       output_dir = "./trained_models/", learning_rate = 1e-3, nb_augmentations = 0,
                       train_to_val_ratio = 0.2):

    if dataset_training is None:
        sys.exit("The input training dataset needs to be defined")
    if output_dir is None:
        sys.exit("The output directory for trained classifier needs to be defined")
    
    os.makedirs(name=output_dir, exist_ok=True)
    file_name_save = os.path.join(output_dir, model_name + ".h5")
    logdir = "logs/scalars/" + model_name
    print(logdir)
    tensorboard_callback = TensorBoard(log_dir=logdir)
    reduce_lr_on_plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001)


    # determine the number of channels and classes
    input_shape = model.layers[0].output_shape
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    train_dict, (X_test, Y_test) = get_data_sample(dataset_training, dataset_validation, nb_channels = n_channels, nb_classes = n_classes, imaging_field_x = imaging_field_x, imaging_field_y = imaging_field_y, nb_augmentations = nb_augmentations, validation_training_ratio = train_to_val_ratio)

    # data information (one way for the user to check if the training dataset makes sense)
    print((nb_augmentations+1)*len(train_dict["channels"]), 'training images')
    print(len(X_test), 'validation images')

    # convert class vectors to binary class matrices
    #train_dict["labels"] = np_utils.to_categorical(train_dict["labels"], n_classes)
    #Y_test = np_utils.to_categorical(Y_test, n_classes).astype('float32')

    # prepare the model compilation
    optimizer = SGD(learning_rate = learning_rate, decay = 1e-07, momentum = 0.9, nesterov = True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # prepare the generation of data
    if nb_augmentations == 0:
        train_generator = random_sample_generator(train_dict["channels"], train_dict["labels"], batch_size, n_channels, n_classes, imaging_field_x, imaging_field_y) 
    else:
        train_generator = random_sample_generator_dataAugmentation(train_dict["channels"], train_dict["labels"], batch_size, n_channels, n_classes, imaging_field_x, imaging_field_y, nb_augmentations) 
        
    # fit the model
    lr_sched = rate_scheduler(lr = learning_rate, decay = 0.95)
    loss_history = model.fit(train_generator,
                                       steps_per_epoch = int((nb_augmentations+1)*len(train_dict["labels"])/batch_size), 
                                       epochs=n_epoch, validation_data=(X_test,Y_test), 
                                       callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto',save_weights_only = True), reduce_lr_on_plateau_callback, tensorboard_callback])


"""
Executing convnets
"""

def get_image_sizes(data_location):
    width = 256
    height = 256
    nb_channels = 1
    img_list = []
    img_list += [getfiles(data_location)]
    img_temp = get_image(os.path.join(data_location, img_list[0][0]))
    if len(img_temp.shape)>2:
        if img_temp.shape[0]<img_temp.shape[2]:
            nb_channels = img_temp.shape[0]
            width = img_temp.shape[1]
            height=img_temp.shape[2]
        else:
            nb_channels = img_temp.shape[2]
            width = img_temp.shape[0]
            height=img_temp.shape[1]
    else:
        width = img_temp.shape[0]
        height=img_temp.shape[1]
    return width, height, nb_channels

def get_images_from_directory(data_location):
    img_list = []
    img_list += [getfiles(data_location)]

    all_images = []
    for stack_iteration in range(len(img_list[0])):
        current_img = get_image(os.path.join(data_location, img_list[0][stack_iteration]))
        all_channels = np.zeros((1, current_img.shape[0], current_img.shape[1], current_img.shape[2]), dtype = 'float32')
        all_channels[0, :, :, :] = current_img
        all_images += [all_channels]
            
    return all_images

def run_model(img, model, imaging_field_x = 256, imaging_field_y = 256):
    
    img[0, :, :, :] = process_image(img[0, :, :, :])
    img = np.pad(img, pad_width = [(0,0), (5,5), (5,5), (0,0)], mode = 'reflect')
            
    n_classes = model.layers[-1].output_shape[-1]
    image_size_x = img.shape[1]
    image_size_y = img.shape[2]
    model_output = np.zeros((n_classes,image_size_x-10,image_size_y-10), dtype = np.float32)
    current_output = np.zeros((1,imaging_field_x,imaging_field_y,n_classes), dtype = np.float32)
    
    x_iterator = 0
    y_iterator = 0
    
    while x_iterator<=(image_size_x-imaging_field_x) and y_iterator<=(image_size_y-imaging_field_y):
        current_output = model.predict(img[:,x_iterator:(x_iterator+imaging_field_x),y_iterator:(y_iterator+imaging_field_y),:])
        for y in range(y_iterator,(y_iterator+imaging_field_y-10)):
            for x in range(x_iterator,(x_iterator+imaging_field_x-10)):
                model_output[:,x,y] = current_output[0, :]
        
        if x_iterator<(image_size_x-2*imaging_field_x):
            x_iterator += (imaging_field_x-10)
        else:
            if x_iterator == (image_size_x-imaging_field_x):
                if y_iterator < (image_size_y-2*imaging_field_y):
                    y_iterator += (imaging_field_y-10)
                    x_iterator = 0
                else:
                    if y_iterator == (image_size_y-imaging_field_y):
                        y_iterator += (imaging_field_y-10)
                    else:
                        y_iterator = (image_size_y-imaging_field_y)
                        x_iterator = 0
            else:
                x_iterator = image_size_x-(imaging_field_x)

    return model_output


def run_models_on_directory(data_location, output_location, model, score, imaging_field_x, imaging_field_y, n_channels, n_classes):

    # create output folder if it doesn't exist
    os.makedirs(name=output_location, exist_ok=True)
    
    # determine the image size
    image_size_x, image_size_y, nb_channels = get_image_sizes(data_location)
    
    if image_size_x<imaging_field_x:
        sys.exit("The input image has a smaller x dimension than the imaging field")
    if image_size_y<imaging_field_y:
        sys.exit("The input image has a smaller y dimension than the imaging field")
    if n_channels!=nb_channels:
        sys.exit("The input image has a different number of channels than indicated in the U-Net architecture")


    # process images
    counter = 0
    img_list_files = [getfiles(data_location)]

    image_list = get_images_from_directory(data_location)

    for img in image_list:
        print("Processing image ",str(counter + 1)," of ",str(len(image_list)))
        processed_image = run_model(img, model, imaging_field_x = imaging_field_x, imaging_field_y = imaging_field_y)
        
        if score==False:
            scores = np.zeros((n_classes))
            for i in range(processed_image.shape[0]):
                scores[i] = np.average(processed_image[i, :, :])
            # Save file
            cnnout_name = os.path.join(output_location, os.path.splitext(img_list_files[0][counter])[0] + ".csv")
            np.savetxt(cnnout_name, scores, delimiter=",")
        else:
            # Save images
            cnnout_name = os.path.join(output_location, os.path.splitext(img_list_files[0][counter])[0] + ".tiff")
            tiff.imsave(cnnout_name, processed_image)

        counter += 1
