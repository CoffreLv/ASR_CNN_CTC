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
model_Path = './acoustic_model/cnn3ctc20190917_1734/e_44.model'
model_session = Acoustic_model(datapath)
model_session.Load_Model(filename = model_Path)
model_session.Test_model_all(modelpath = model_Path, datapath = './dataset', str_Data = 'train', data_Count = 500 )
