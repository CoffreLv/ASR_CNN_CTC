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
        #self.label_max_length = 128  #？？？
        self.AUDIO_LENGTH = 1600    #一次送入的特征个数
        self.AUDIO_FEATURE_LENGTH = 200 #每个特征的维度
        self._model, self.base_model = self.Create_model()  #建立模型并返回模型

        self.datapath = datapath    #初始化数据路径赋值
        self.slash  = '/'
        self.datapath = self.datapath + self.slash
        self.now_Time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))

    def Create_model(self): #卷积层*6
        '''
        定义模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小维3*3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE,使用softmax作为激活函数
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出
        '''

        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))

        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2d_1')(input_data)  #卷积层1
        layer_c1 = Dropout(0.05)(layer_c1)   #为卷积层1添加Dropout
        layer_c2 = Conv2D(32, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2d_2')(layer_c1)    #卷积层2
        layer_p3 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid', name = 'max_pooling2d_1_tr')(layer_c2) #池化层3
        layer_p3 = Dropout(0.05)(layer_p3)
        layer_c4 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2d_3')(layer_p3)    #卷积层4
        layer_c4 = Dropout(0.1)(layer_c4)   #为卷积层4添加Dropout
        layer_c5 = Conv2D(64, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2d_4')(layer_c4)    #卷积层5
        layer_p6 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid', name = 'max_pooling2d_2_tr')(layer_c5) #池化层6
        layer_p6 = Dropout(0.1)(layer_p6)
        layer_c7 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2d_5_tr')(layer_p6)    #卷积层7
        #layer_c7 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2d_5')(layer_p6)    #卷积层7
        layer_c7 = Dropout(0.15)(layer_c7)   #为卷积层7添加Dropout
        layer_c8 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2d_6_tr')(layer_c7)    #卷积层8
        #layer_c8 = Conv2D(128, (3, 3), use_bias = True, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2d_6')(layer_c7)    #卷积层8
        layer_p9 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid', name = 'max_pooling2d_3_tr')(layer_c8) #池化层9
        #修改音频长度需要对应修改
        layer_f10 = Reshape((200, 3200),name = 'reshape_1')(layer_p9)  #Reshape层10
        layer_f10 = Dropout(0.2)(layer_f10)
        layer_f11 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f10)    #全连接层11
        layer_f11 = Dropout(0.3)(layer_f11)
        #layer_fu12 = Dense(1424 , use_bias = True, kernel_initializer = 'he_normal')(layer_f11)
        layer_fu12_tr = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal', name = 'dense_2_tr')(layer_f11)
################################################################################################
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu12_tr)
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
        print("训练数据条数：%d"%num_Data)
        filepath = './acoustic_model/' + model_Name + self.now_Time + '/'
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        #check_Point = kr.callbacks.ModelCheckpoint(filepath + 'e_{epoch:02d}.model', monitor = 'val_loss', verbose = 2, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)  #每个epoch保存模型
        #early_Stopping = kr.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 1, mode = 'auto')  #在训练过程中monitor = val_loss值patience轮不下降 min_delta 停止训练
        check_Point = kr.callbacks.ModelCheckpoint(filepath + 'e_{epoch:02d}.model', monitor = 'loss', verbose = 2, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)  #每个epoch保存模型
        early_Stopping = kr.callbacks.EarlyStopping(monitor = 'loss', min_delta = 0, patience = 10, verbose = 1, mode = 'auto')  #在训练过程中monitor = loss值patience轮不下降 min_delta 停止训练
        self._model.fit_generator(data_gentator, steps_per_epoch = 900, epochs = epoch, callbacks = [check_Point, early_Stopping], validation_data = validation_Data_Gentator)

    def Load_Model(self, filename = abspath + 'acoustic_model/' + model_Name , comment = ''):   #加载模型参数
        self._model.load_weights(filename, by_name = True)
        f_training = open(filepath + 'load_model_information.txt', mode = 'w', encoding = 'utf-8')    #载入模型信息留存
        f_training.write("载入模型路径：" + filename + '\n')
        f_training.close()

    def Print_layer_name(self):
        num_Of_Layers = len(self.model.layers) - 4  #减4是因为去掉了CTC和label,input_Length,label_length层
        layers_Output = {}
        for i in range(num_Of_Layers):
            name_Of_Output_Layer = self.model.layers[i].name
            print(name_Of_Output_Layer)
        return layers_Output

    @property
    def model(self):
        '''
        返回keras model
        '''
        return self._model

if(__name__ == '__main__'):
    exit()
