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

session_hand = acoustic_model.Acoustic_model(datapath)
session_hand.Load_Model(filename = './acoustic_model/cnn3ctc20190412_0819/e_1035.model')
session_hand.Model_training_all(datapath)
