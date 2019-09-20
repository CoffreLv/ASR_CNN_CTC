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

    def Create_model(self): #卷积层*3
        input_data = Input(name = 'the_input', shape = (self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH,1))
        layer_c1 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_data)  #卷积层1
        layer_c1 = Dropout(0.05)(layer_c1)   #为卷积层1添加Dropout
        layer_c2 = Conv2D(32, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c1)  #卷积层2
        layer_p1 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c2)
        layer_p1 = Dropout(0.05)(layer_p1)

        layer_c3 = Conv2D(64, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p1)  #卷积层1
        layer_c3 = Dropout(0.1)(layer_c3)   #为卷积层1添加Dropout
        layer_c4 = Conv2D(64, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c3)  #卷积层2
        layer_p2 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c4)
        layer_p2 = Dropout(0.1)(layer_p2)

        layer_c5 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p2)  #卷积层1
        layer_c5 = Dropout(0.15)(layer_c5)   #为卷积层1添加Dropout
        layer_c6 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c5)  #卷积层2
        layer_p3 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid')(layer_c6)
        layer_p3 = Dropout(0.15)(layer_p3)

        layer_c7 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p3)  #卷积层1
        layer_c7 = Dropout(0.2)(layer_c7)   #为卷积层1添加Dropout
        layer_c8 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c7)  #卷积层2
        layer_p4 = MaxPooling2D(pool_size = 1, strides = None, padding = 'valid')(layer_c8)
        layer_p4 = Dropout(0.2)(layer_p4)

        layer_c9 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p4)  #卷积层1
        layer_c9 = Dropout(0.3)(layer_c9)   #为卷积层1添加Dropout
        layer_c10 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c9)  #卷积层2
        layer_p5 = MaxPooling2D(pool_size = 1, strides = None, padding = 'valid')(layer_c10)
        layer_p5 = Dropout(0.3)(layer_p5)

        layer_c11 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_p5)  #卷积层1
        layer_c11 = Dropout(0.4)(layer_c11)   #为卷积层1添加Dropout
        layer_c12 = Conv2D(128, (3, 3), use_bias = False, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer_c11)  #卷积层2
        layer_p6 = MaxPooling2D(pool_size = 1, strides = None, padding = 'valid')(layer_c12)
        layer_p6 = Dropout(0.4)(layer_p6)

        layer_f7 = Reshape((200, 3200))(layer_p6)  #Reshape#修改输入维度需要对应修改
        layer_f7 = Dropout(0.2)(layer_f7)
        layer_f8 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f7)    #全连接层8
        layer_f8 = Dropout(0.3)(layer_f8)
        layer_f9 = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal')(layer_f8)
        y_pre = Activation('softmax', name = 'Activation')(layer_f9)
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
        num_Data_Dev = validation_Data_Gentator.list_Datas  #获取验证集数量
        num_Data = data_gentator.list_Datas  #获取数据数量
        print("训练数据条数：%d"%num_Data)
        filepath = './acoustic_model/' + model_Name + self.now_Time + '/'
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        f_training = open(filepath + 'training_information.txt', mode = 'w', encoding = 'utf-8')    #训练信息留存
        f_training.write("训练数据条数：" + str(num_Data) + '\n')
        f_training.write("验证数据条数：" + str(num_Data_Dev) + '\n')
        f_training.write("网络结构：" + str(self.model.layers) + '\n')
        f_training.close()
        #check_Point = kr.callbacks.ModelCheckpoint(filepath + 'e_{epoch:02d}.model', monitor = 'val_loss', verbose = 2, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)  #每个epoch保存模型
        #early_Stopping = kr.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 1, mode = 'auto')  #在训练过程中monitor = val_loss值patience轮不下降 min_delta 停止训练
        check_Point = kr.callbacks.ModelCheckpoint(filepath + 'e_{epoch:02d}.model', monitor = 'loss', verbose = 2, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)  #每个epoch保存模型
        early_Stopping = kr.callbacks.EarlyStopping(monitor = 'loss', min_delta = 0, patience = 10, verbose = 1, mode = 'auto')  #在训练过程中monitor = loss值patience轮不下降 min_delta 停止训练
        self._model.fit_generator(data_gentator, steps_per_epoch = 700, epochs = epoch, callbacks = [check_Point, early_Stopping], validation_data = validation_Data_Gentator)

    def Load_Model(self, filename = abspath + 'acoustic_model/' + model_Name , comment = ''):   #加载模型参数
        self._model.load_weights(filename)
        file_Path = filename.split('/')
        filepath = '/'.join(file_Path[:-1]) + '/'
        print(filepath)
        f_training = open(filepath + 'load_model_information.txt', mode = 'a+', encoding = 'utf-8')    #载入模型信息留存
        f_training.write("载入模型路径：" + filename + '\n')
        f_training.close()

    def Test_model_all(self, modelpath , datapath = '', str_Data = 'dev', data_Count = 100, out_Report = True, show_Ratio = True, io_Step_Print = 10, io_Step_File = 10):
        #测试检验模型效果
        data = Acoustic_data(self.datapath , str_Data)
        num_Data = data.Get_data_num() # 获取数据的数量
        if(data_Count <= 0 or data_Count > num_Data): # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_Count = num_Data
        try:
            words_Num = 0
            word_Error_Num = 0
            nowtime = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
            if(out_Report == True):
                logpath = modelpath + '_' + str_Data + '/'
                if not os.path.exists(logpath):
                    os.mkdir(logpath)
                f = open(logpath + nowtime + '.txt', 'w', encoding='UTF-8') # 打开文件并读入
            tmp = '测试报告\n模型编号 ' + model_Name + '\n\n'
            for i in range(data_Count):
                data_Input, data_Labels = data.Get_data_all( i )  # 从随机数开始连续向后取一定数量数据
                # 数据格式出错处理 开始
                # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
                num_Bias = 0
                while(data_Input.shape[0] > self.AUDIO_LENGTH):
                    print('*[Error]','wave data lenghth of num', str(i) , 'is too long.','\n A Exception raise when test Speech Model.')
                    num_Bias += 1
                    data_Input, data_Labels = data.Get_data(i + num_Bias)  # 从随机数开始连续向后取一定数量数据
                    #=data_Input, data_Labels = data.Get_data(1, data_Count)  # 从随机数开始连续向后取一定数量数据
                pre = self.Predict(data_Input, data_Input.shape[0] // 8)

                #预测对比字符输出准备
                list_Symbol_Dict = get_feature.Get_symbol_list()

                r_PRE = []  #初始化列表用于存储预测对应字符串
                for i in pre:
                    r_PRE.append(list_Symbol_Dict[i])
                r_NOW = []  #初始化列表用于存储原字符串
                for j in data_Labels:
                    r_NOW.append(list_Symbol_Dict[j])

                words_Num_Now = data_Labels.shape[0] # 获取每个句子的字数
                words_Num += words_Num_Now # 把句子的总字数加上
                edit_Distance = self.Get_edit_distance(data_Labels, pre) # 获取编辑距离
                if(edit_Distance <= words_Num): # 当编辑距离小于等于句子字数时
                    word_Error_Num += edit_Distance # 使用编辑距离作为错误字数
                else: # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                    word_Error_Num += words_Num_Now # 就直接加句子本来的总字数就好了
                    if((i % io_Step_Print == 0 or i == data_Count - 1) and show_Ratio == True):
                        print('Test Count: ',i,'/',data_Count)
                if(out_Report == True):
                    if(i % io_Step_File == 0 or i == data_Count - 1):
                        f.write(tmp)
                        tmp = ''
                    tmp += str(i) + '\n'
                    tmp += 'True:\t' + str(data_Labels) + '\n'
                    tmp += '\t' + str(r_NOW) + '\n'
                    tmp += 'Pred:\t' + str(pre) + '\n'
                    tmp += '\t' + str(r_PRE) + '\n'
                    tmp += '\n'
                print('*[Test Result] Speech Recognition ' + str_Data + ' set word error ratio: ', word_Error_Num / words_Num * 100, '%')
            if(out_Report == True):
                tmp += '*[测试结果] 语音识别 ' + str_Data + ' 集语音单字错误率： ' + str(word_Error_Num / words_Num * 100) + ' %'
                f.write(tmp)
                tmp = ''
                f.close()
        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

    def Layer_output(self, datapath = '', str_Data = 'dev', data_Count = 1): #输出中间层特征
        data = Acoustic_data(self.datapath , str_Data)
        num_Data = data.Get_data_num() # 获取数据的数量
        if(data_Count <= 0 or data_Count > num_Data): # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_Count = num_Data
        for i in range(data_Count):
            data_Input, data_Labels = data.Get_data_all( i )  # 从随机数开始连续向后取一定数量数据
            # 数据格式出错处理 开始
            # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
            num_Bias = 0
            while(data_Input.shape[0] > self.AUDIO_LENGTH):
                print('*[Error]','wave data lenghth of num', str(i) , 'is too long.','\n A Exception raise when test Speech Model.')
                num_Bias += 1
                data_Input, data_Labels = data.Get_data(i + num_Bias)  # 从随机数开始连续向后取一定数量数据
        batch_size = 1
        in_len = np.zeros((batch_size),dtype = np.int32)
        in_len[0] = data_Input.shape[0] // 8
        x_in = np.zeros((batch_size, 1600, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
        for i in range(batch_size):
            x_in[i,0:len(data_Input)] = data_Input

        layers_Output = self.Get_mid_layer_output(x_in) #定义层输出字典
        return layers_Output

    def Get_mid_layer_output(self, x_in):
        num_Of_Layers = len(self.model.layers) - 4  #减4是因为去掉了CTC和label,input_Length,label_length层
        layers_Output = {}
        for i in range(num_Of_Layers):
            name_Of_Output_Layer = self.model.layers[i].name
            get_Layer_Output = BK.function([self.model.layers[0].input, BK.learning_phase()], [self.model.layers[i].output])
            layers_Output[name_Of_Output_Layer] = get_Layer_Output([x_in, 0])[0]
        return layers_Output

    def Predict(self, data_Input, input_len):
        '''
        预测结果
        返回语音识别后的拼音符号列表
        '''
        batch_size = 1
        in_len = np.zeros((batch_size),dtype = np.int32)
        in_len[0] = input_len
        x_in = np.zeros((batch_size, 1600, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
        for i in range(batch_size):
            x_in[i,0:len(data_Input)] = data_Input
        base_Pred = self.base_model.predict(x = x_in)
        base_Pred = base_Pred[:, :, :]
        r = BK.ctc_decode(base_Pred, in_len, greedy = True, beam_width=100, top_paths=1)
        r1 = BK.get_value(r[0][0])
        r1=r1[0]
        return r1
        pass

    def speech_Recognize(self, wav_Signal, fs): #识别函数       有待修改
        data_Input = get_feature.Get_frequecy_feature(wav_Signal, fs)
        input_Length = len(data_Input)
        input_Length = input_Length // 8

        data_Input = np.array(data_Input, dtype = np.float)
        data_Input = data_Input.reshape(data_Input.shape[0], data_Input.shape[1], 1)
        r1 = self.Predict(data_Input, input_Length)
        list_Symbol_Dict = get_feature.Get_symbol_list(self.datapath)

        r_STR = []
        for i in r1:
            r_STR.append(list_Symbol_Dict[i])
        return r_STR
        pass

    def speech_Recognize_Fromfile(self, filename):
        wav_Signal, fs = Read_wav_data(filename)
        r = self.speech_Recognize(wav_Signal, fs)
        return r
        pass

    def Get_edit_distance(self, str1, str2):
        leven_cost = 0
        s = difflib.SequenceMatcher(None, str1, str2)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == 'replace':
                leven_cost += max(i2-i1, j2-j1)
            elif tag == 'insert':
                leven_cost += (j2-j1)
            elif tag == 'delete':
                leven_cost += (i2-i1)
        return leven_cost

    @property
    def model(self):
        '''
        返回keras model
        '''
        return self._model

if(__name__ == '__main__'):
    exit()
