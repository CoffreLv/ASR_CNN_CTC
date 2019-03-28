#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified: 2019-03-19 10:54
# Email        : coffrelv@163.com
# Filename     : get_data_generation.py
# Description  : 
# ******************************************************
import numpy as np
import keras

import get_feature
from get_feature import Acoustic_data

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_Path, data_Type, batch_Size =32,  shuffle=True):
        self.batch_Size = batch_Size
        self.data_Path = data_Path
        self.data_Type = data_Type
        self.acoustic_Data = Acoustic_data( self.data_Path, self.data_Type)
        self.list_Datas = self.acoustic_Data.Get_data_num()
        self.shuffle = shuffle
        self.audio_length = 1600
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.list_Datas / self.batch_Size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_Size:(index+1)*self.batch_Size]

        # Find list of IDs
        list_Datas_Temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_Datas_Temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.list_Datas)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_Datas_Temp):
        # Initialization
        X = np.zeros((self.batch_Size, self.audio_length, 200, 1), dtype = np.float)
        y = np.zeros((self.batch_Size, 64), dtype = np.int16)

        # Generate data_Type
        labels = []
        for z in range(0, self.batch_Size):
            labels.append([0.0])
        labels = np.array(labels, dtype = np.float)
        input_Length = []
        label_Length = []
        for i,j in enumerate(list_Datas_Temp):
            # Store sample
            data_Input , data_Labels = self.acoustic_Data.Get_data_all(num_Start = j)
            input_Length.append(data_Input.shape[0] // 8 + data_Input.shape[0] % 8)
            X[i,0:len(data_Input)] = data_Input
            y[i,0:len(data_Labels)] = data_Labels
            label_Length.append([len(data_Labels)])
        label_Length = np.matrix(label_Length)
        input_Length = np.array(input_Length).T
        return [X,y,input_Length, label_Length], labels
