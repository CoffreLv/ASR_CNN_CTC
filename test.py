#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified: 2018-12-18 15:51
# Email        : coffrelv@163.com
# Filename     : test.py
# Description  : 
# ******************************************************

from acoustic_model import Acoustic_model

datapath = 'dataset'

model_session = Acoustic_model(datapath)
model_session.Load_Model(filename = './acoustic_model/cnn3ctc20190328_1851/e_02.model')
model_session.Test_model_all(datapath = './dataset', str_Data = 'test', data_Count = 1200)
