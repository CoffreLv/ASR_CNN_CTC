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
session_hand.Load_Model(filename = './acoustic_model/2019-09-04_EN_77/e_77.model')
session_hand.Model_training_all(datapath)
