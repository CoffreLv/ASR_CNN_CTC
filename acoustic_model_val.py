#/usr/bin/python
# ******************************************************
# Author       : CoffreLv
# Last modified:	2019-03-29 08:50
# Email        : coffrelv@163.com
# Filename     :	acoustic_model.py
# Description  :    声学模型类 
# ******************************************************

import os
import time
import random
import difflib

import keras as kr
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape
from keras.layers import Activation, Conv2D, MaxPooling2D, Lambda
from keras.optimizers import Adam
from keras import backend as BK

import get_feature
import get_data_generation
from get_feature import Acoustic_data
from get_data_generation import DataGenerator
abspath = ''
model_Name = 'cnn3ctc'

class Acoustic_model(): #声学模型类
    def __init__(self , datapath):
        '''
        初始化
        '''
        MS_OUTPUT_SIZE = 973
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE    #模型最终输出的维度大小
        self.label_max_length = 64  #？？？
        self.AUDIO_LENGTH = 1600    #一次送入的特征个数
        self.AUDIO_FEATURE_LENGTH = 200 #每个特征的维度
        self._model, self.base_model = self.Create_model()  #建立模型并返回模型

        self.datapath = datapath    #初始化数据路径赋值
        self.slash  = '/'
        self.datapath = self.datapath + self.slash
        self.now_Time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))

#二层卷积层
    def Create_model(self): #卷积层*3
        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))

        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_data)  #卷积层1
        layer_c1 = Dropout(0.1)(layer_c1)   #为卷积层1添加Dropout
        layer_c2 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c1)  #卷积层2
        layer_c2 = Dropout(0.1)(layer_c1)
        layer_p1 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c2)
        layer_p1 = Dropout(0.1)(layer_p1)
        layer_c3 = Conv2D(64, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(p1)  #卷积层1
        layer_c3 = Dropout(0.2)(layer_c3)   #为卷积层1添加Dropout
        layer_c4 = Conv2D(64, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c3)  #卷积层2
        layer_c4 = Dropout(0.2)(layer_c4)
        layer_p2 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c4)
        layer_p2 = Dropout(0.2)(layer_p2)
        layer_c5 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p2)  #卷积层1
        layer_c5 = Dropout(0.3)(layer_c5)   #为卷积层1添加Dropout
        layer_c6 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c5)  #卷积层2
        layer_c6 = Dropout(0.3)(layer_c6)
        layer_p3 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c6)
        layer_p3 = Dropout(0.3)(layer_p3)
        layer_c7 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p3)  #卷积层1
        layer_c7 = Dropout(0.5)(layer_c7)   #为卷积层1添加Dropout
        layer_c8 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c7)  #卷积层2
        layer_c8 = Dropout(0.5)(layer_c8)
        layer_p4 = MaxPooling2D(pool_size = 1, strides = None, padding = 'valid')(layer_c8)
        #layer_p1 = Dropout(0.1)(layer_p1)
        #修改音频长度需要对应修改
        layer_f7 = Reshape((200, 51200))(layer_p4)  #Reshape
        layer_f7 = Dropout(0.2)(layer_f7)
        layer_f8 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f7)    #全连接层8
        layer_f8 = Dropout(0.3)(layer_f8)
        layer_fu9 = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal')(layer_f8)
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu9)
        model_data = Model(inputs = input_data, output = y_pre)

        labels = Input(name = 'the_labels', shape = [self.label_max_length], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape = (1, ), name = 'ctc')([y_pre,labels, input_length, label_length])

        model = Model(inputs = [input_data, labels, input_length, label_length], output = loss_out)
        model.summary()
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss = {'ctc': lambda y_true, y_pre: y_pre}, optimizer = opt)

        print('[Info]创建编译模型成功')
        return model, model_data

    def ctc_lambda_func(self, args):
        y_pre , labels, input_length, label_length = args

        y_pre = y_pre[:,:,:]
        return BK.ctc_batch_cost(labels, y_pre, input_length, label_length)

    def Model_training_all(self, datapath, epoch = 10000, batch_Size = 8): #抽取全部数据训练
        '''
        训练模型
        参数：
                datapath:数据路径
                epoch:迭代轮数
        '''
        data_gentator = DataGenerator(batch_Size = batch_Size, data_Path = datapath, data_Type = 'train')   #生成训练数据迭代器
        validation_Data_Gentator = DataGenerator(batch_Size = batch_Size, data_Path = datapath, data_Type = 'dev')  #生成验证数据迭代器
        num_Data = data_gentator.list_Datas  #获取数据数量
        num_Data_Dev = validation_Data_Gentator.list_Datas  #获取验证集数量
        print("训练数据条数：%d"%num_Data)
        filepath = './acoustic_model/' + model_Name + self.now_Time + '/'
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        f_training = open(filepath + 'training_information.txt', mode = 'w', encoding = 'utf-8')    #训练信息留存
        f_training.write("训练数据条数：" + str(num_Data) + '\n')
        f_training.write("验证数据条数：" + str(num_Data_Dev) + '\n')
        layers_Name = self.Get_layers_name()
        f_training.write("网络结构：" + str(layers_Name) + '\n')
        f_training.close()
        check_Point = kr.callbacks.ModelCheckpoint(filepath + 'e_{epoch:02d}.model', monitor = 'val_loss', verbose = 2, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)  #每个epoch保存模型
        early_Stopping = kr.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 1, mode = 'auto')  #在训练过程中monitor = val_loss值patience轮不下降 min_delta 停止训练
        self._model.fit_generator(data_gentator, steps_per_epoch = 900, epochs = epoch, callbacks = [check_Point, early_Stopping], validation_data = validation_Data_Gentator)

    def Get_layers_name(self):
        num_Of_Layers = len(self.model.layers)
        name_Of_Output_Layer = []
        for i in range(num_Of_Layers):
            name_Of_Output_Layer.append(self.model.layers[i].name)
        return name_Of_Output_Layer

    @property
    def model(self):
        '''
        返回keras model
        '''
        return self._model

if(__name__ == '__main__'):
    exit()
