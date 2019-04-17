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
import prepare_dataset
import build_dataset

datapath = 'dataset'
modelpath = 'acoustic_model/'

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if (not os.path.exists(modelpath)): #创建模型存储目录
    os.makedirs(modelpath)

prepare_dataset.Main_self('train', 'cv', 'test')
build_dataset.Main_self()
model_session = Acoustic_model(datapath)
model_session.Model_training_all(datapath)
