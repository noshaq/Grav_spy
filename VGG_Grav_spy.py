import os
import time
import numpy as np
import pandas as pd
from glob import glob
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Convolution2D, MaxPooling2D,Input
from keras.layers import Dense, Dropout, Activation, Flatten,MaxPool2D
from keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalMaxPooling3D, Dropout, BatchNormalization 

import matplotlib.pyplot as plt

plt.switch_backend('agg')

np.random.seed(4321)



def VGG(train_generator, validation_generator, test_generator, nb_classes, channels):
    '''
    Define the Convolutional Neural Network

    INPUT
        train_generator:  generator for reading train data from folder
        validation_generator: idem
        test_generator: idem
        channels: Specify if the image is grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification
'''
    img_rows = train_generator.target_size[0]
    img_cols = train_generator.target_size[0]

    input_shape = (img_rows, img_cols, channels) #  (height, width, channel RGB)

    #
    model = Sequential(name='vgg16')

    # block1
    model.add(Conv2D(64, (channels, channels), padding='same', activation='relu', input_shape=input_shape, name='block1_conv1'))
    model.add(Conv2D(64, (channels, channels), padding='same', activation='relu', name='block1_conv2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block1_pool'))
    model.add(BatchNormalization())


    # block2
    model.add(Conv2D(128, (channels, channels), padding='same', activation='relu', name='block2_conv1'))
    model.add(Conv2D(128, (channels, channels), padding='same', activation='relu', name='block2_conv2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block2_pool'))
    model.add(BatchNormalization())

    # block3
    model.add(Conv2D(256, (channels, channels), padding='same', activation='relu', name='block3_conv1'))
    model.add(Conv2D(256, (channels, channels), padding='same', activation='relu', name='block3_conv2'))
    model.add(Conv2D(256, (channels, channels), padding='same', activation='relu', name='block3_conv3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block3_pool'))
    model.add(BatchNormalization())

    # block4
    model.add(Conv2D(512, (channels, channels), padding='same', activation='relu', name='block4_conv1'))
    model.add(Conv2D(512, (channels, channels), padding='same', activation='relu', name='block4_conv2'))
    model.add(Conv2D(512, (channels, channels), padding='same', activation='relu', name='block4_conv3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block4_pool'))
    model.add(BatchNormalization())

    # block5
    model.add(Conv2D(512, (channels, channels), padding='same', activation='relu', name='block5_conv1'))
    model.add(Conv2D(512, (channels, channels), padding='same', activation='relu', name='block5_conv2'))
    model.add(Conv2D(512, (channels, channels), padding='same', activation='relu', name='block5_conv3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block5_pool'))
    model.add(BatchNormalization())

    

    # Classification
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fully_c1'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu', name='fully_c2'))
    model.add(Dense(nb_classes, activation='softmax', name='Predicted_classes'))

    # show me the network!!!
    model.summary()
  


    

    return model
