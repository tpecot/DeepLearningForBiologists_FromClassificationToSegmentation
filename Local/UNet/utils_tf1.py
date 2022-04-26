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
import os

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import SGD, RMSprop
    
import ipywidgets as widgets
import ipyfilechooser
from ipyfilechooser import FileChooser
from ipywidgets import HBox, Label, Layout

from models import unet
import tempfile
import shutil

"""
Interfaces
"""


def saving_model_for_Fiji_plugin_interface(nb_trainings):
    input_classifier = np.zeros([nb_trainings], FileChooser)
    output_folder = np.zeros([nb_trainings], FileChooser)
    output_name = np.zeros([nb_trainings], FileChooser)
    nb_channels = np.zeros([nb_trainings], HBox)
    nb_classes = np.zeros([nb_trainings], HBox)
    imaging_field_x = np.zeros([nb_trainings], HBox)
    imaging_field_y = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Input model")
        input_classifier[i] = FileChooser('./models')
        display(input_classifier[i])
        print('\x1b[1m'+"Output directory")
        output_folder[i] = FileChooser('./models')
        display(output_folder[i])

        label_layout = Layout(width='200px',height='30px')
        
        output_name[i] = HBox([Label('Output name:', layout=label_layout), widgets.Text(
            value='', description='',disabled=False)])
        display(output_name[i])

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

        
    parameters.append(input_classifier)
    parameters.append(output_folder)
    parameters.append(output_name)
    parameters.append(nb_channels)
    parameters.append(nb_classes)
    parameters.append(imaging_field_x)
    parameters.append(imaging_field_y)
        
    return parameters  
        

def saving_model_for_Fiji_plugin(nb_runnings, parameters):
    for i in range(nb_runnings):
        if parameters[0][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select a trained model to run your images")
        if parameters[1][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an output directory")

        model_path = parameters[0][i].selected
        nb_classes = parameters[4][i].children[1].value
        dim_x = parameters[5][i].children[1].value
        dim_y = parameters[6][i].children[1].value
        nb_channels = parameters[3][i].children[1].value
        model = unet(nb_classes, dim_x, dim_y, nb_channels, model_path)
        save_path = parameters[1][i].selected+parameters[2][i].children[1].value

        tmp_path = tempfile.TemporaryDirectory()

        signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={'image': model.input}, outputs={'scores': model.output})
        builder = tf.saved_model.builder.SavedModelBuilder(os.path.abspath(tmp_path.name))
        builder.add_meta_graph_and_variables(
            sess=K.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
            })
    
        verbose = builder.save()
        verbose = shutil.copytree(tmp_path.name, save_path)
        verbose = shutil.make_archive(save_path, "zip", save_path, "./")
        for filename in os.listdir(save_path):
            file_path = os.path.join(save_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        shutil.rmtree(save_path)
        
        del model
        

