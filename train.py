#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified: 2018-12-18 15:51
# Email        : coffrelv@163.com
# Filename     : train.py
# Description  : 
# ******************************************************

import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from acoustic_model import Acoustic_model

datapath = 'dataset'
modelpath = 'acoustic_model/'

if (not os.path.exists(modelpath)): #创建模型存储目录
    os.makedirs(modelpath)

model_session = Acoustic_model(datapath)
model_session.Model_training_All(datapath, epoch = 100, batch_size = 8)
