#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified: 2019-04-17 11:14
# Email        : coffrelv@163.com
# Filename     : train_with_model.py
# Description  : 
# ******************************************************

import acoustic_model

datapath = 'dataset'

model_Path = './acoustic_model/cnn3ctc20190912_1609/e_152.model'
model_session = acoustic_model.Acoustic_model(datapath)
model_session.Load_Model(filename = model_Path)
model_session.Model_training_all(datapath)
