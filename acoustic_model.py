#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified: 2018-12-18 15:52
# Email        : coffrelv@163.com
# Filename     : acoustic_model.py
# Description  : 
# ******************************************************

import os
import time
import random

import keras as kr
import numpy as np
from keras.models import Model
from keras.layers import Dense,Droput, Input, Reshape
from keras.layers import Activation, Conv2D, MaxPooling2D, Lambda
from keras.optimizers import Adam
from keras import backend as BK

class Acoustic_model(): #声学模型类
    def __init__(self , datapath):
        '''
        初始化
        '''
        MS_OUTPUT_SIZE = 100
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE    #模型最终输出的维度大小
        self.label_max_length = 64  #？？？
        self.AUDIO_LENGTH = 1600    #一次送入的特征个数
        self.AUDIO_FEATURE_LENGTH = 200 #每个特征的维度
        self._model, self.base_model = self.Create_model()  #建立模型并返回模型

        self.datapath = datapath    #初始化数据路径赋值
        self.slash  = '/'
        self.datapath = self.datapath + self.slash

    def Create_model(self):
        '''
        定义模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小维3*3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE,使用softmax作为激活函数
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出
        '''

        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))

        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_data)  #卷积层1
        layer_c1 = Dropout(0.05)(layer_c1)   #为卷积层1添加Dropout
        layer_c2 = Conv2D(32, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c1)    #卷积层2
        layer_p3 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c2) #池化层3
        layer_p3 = Dropout(0.05)(layer_p3)
        layer_c4 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p3)    #卷积层4
        layer_c4 = Dropout(0.1)(layer_c4)   #为卷积层4添加Dropout
        layer_c5 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c4)    #卷积层5
        layer_p6 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c5) #池化层6
        layer_p6 = Dropout(0.1)(layer_p6)
        layer_c7 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p6)    #卷积层7
        layer_c7 = Dropout(0.15)(layer_c7)   #为卷积层7添加Dropout
        layer_c8 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c7)    #卷积层8
        layer_p9 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c8) #池化层9
        layer_f10 = Reshape((200, 3200))(layer_p9)  #Reshape层10
        layer_f10 = Dropout(0.2)(layer_f10)
        layer_fu11 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f10)    #全连接层11
################################################################################################
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu11)
        model_data = Model(inputs = input_data, output = y_pre)

        labels = Input(name = 'the_labels', shape = [self.label_max_length], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape - (1, ), name = 'ctc')([y_pre,labels, input_length, label_length])

        model = Model(inputs = [input_data, labels, input_length, label_length], output = loss_out)
        model.summary()
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss = {'ctc': lambda y_true, y_pre: y_pre}, optimizer = opt)

        test_func = BK.function([input_data],[y_pre])

        print('[Info]创建编译模型成功')
        return model, model_data

    def ctc_lambda_func(self, args):
        y_pre , labels, input_length, label_length = args

        y_pre = y_pre[:,:,:]
        return BK.ctc_batch_cost(labels, y_pre, input_length, label_length)


