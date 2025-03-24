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

from stardist.models import StarDist2D

"""
Interfaces
"""


def saving_model_for_Fiji_plugin_interface(nb_trainings):
    input_classifier = np.zeros([nb_trainings], FileChooser)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Input model")
        input_classifier[i] = FileChooser('./models')
        display(input_classifier[i])
        
    parameters.append(input_classifier)
        
    return parameters  
        

def saving_model_for_Fiji_plugin(nb_runnings, parameters):
    for i in range(nb_runnings):
        if parameters[0][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select a trained model to run your images")

        model_path = parameters[0][i].selected
        model = StarDist2D(None, name = os.path.split(os.path.dirname(model_path))[-1], basedir = os.path.abspath(os.path.join(model_path,os.pardir)))
        model.export_TF()
        
        del model