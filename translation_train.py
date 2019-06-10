#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified: 2019-04-17 11:14
# Email        : coffrelv@163.com
# Filename     : train_with_model.py
# Description  : 
# ******************************************************

import acoustic_model_translation

datapath = 'dataset'

session_hand = acoustic_model_translation.Acoustic_model(datapath)
session_hand.Load_Model(filename = './acoustic_model/e_27.model')
session_hand.Model_training_all(datapath)
