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
import difflib

import keras as kr
import numpy as np
from keras.models import Model
from keras.layers import Dense,Dropout, Input, Reshape
from keras.layers import Activation, Conv2D, MaxPooling2D, Lambda
from keras.optimizers import Adam
from keras import backend as BK
from get_feature import Acoustic_data

abspath = ''
model_Name = 'cnn3ctc'
save_Model_counter = 0

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

        #修改音频长度需要对应修改
        layer_f10 = Reshape((200, 3200))(layer_p9)  #Reshape层10
        layer_f10 = Dropout(0.2)(layer_f10)
        layer_f11 = Dense(128, activation = 'relu', use_bias = True, kernel_initializer = 'he_normal')(layer_f10)    #全连接层11
        layer_f11 = Dropout(0.3)(layer_f11)
        layer_fu12 = Dense(self.MS_OUTPUT_SIZE, use_bias = True, kernel_initializer = 'he_normal')(layer_f11)
################################################################################################
        y_pre = Activation('softmax', name = 'Activation0')(layer_fu12)
        model_data = Model(inputs = input_data, output = y_pre)

        labels = Input(name = 'the_labels', shape = [self.label_max_length], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape = (1, ), name = 'ctc')([y_pre,labels, input_length, label_length])

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

    def Model_training_All(self, datapath, epoch = 1, batch_size = 16): #抽取全部数据训练
        '''
        训练模型
        参数：
                datapath:数据路径
                epoch:迭代轮数
        '''
        now_Time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
        data_Counter = 0
        data = Acoustic_data(datapath, 'train')
        num_Data = data.Get_data_num()  #获取数据数量
        print("训练数据条数：%d"%num_Data)
        for epoch in range(epoch):  #迭代次数
            yield_Datas = data.data_Genetator_All(batch_size, self.AUDIO_LENGTH)
            print('\n[running] train epoch %d .' % epoch)
            while data_Counter*batch_size < num_Data:
                try:
                    self._model.fit_generator(yield_Datas, steps_per_epoch = 25)
                except StopIteration:
                    print('[error] generator error. Please check data format.')
                    break
            self.Save_model(filepath = abspath + 'acoustic_model/' + model_Name + now_Time + '/',comment = 'e_' + str(epoch))
            self.Test_model(self.datapath, str_Data = 'train', data_Count = 10)
            #self.Test_model(self.datapath, str_Data = 'cv', data_Count = 10)

    def Model_training(self, datapath, epoch = 1, save_Step = 1000, batch_size = 16):   #随机抽取数据训练
        '''
        训练模型
        参数：
                datapath:数据路径
                epoch:迭代轮数
                save_step:每多少步保存一次模型
        '''
        now_Time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
        data = Acoustic_data(datapath, 'train')
        num_Data = data.Get_data_num()  #获取数据数量
        yield_Datas = data.Data_genetator(batch_size, self.AUDIO_LENGTH)
        for epoch in range(epoch):  #迭代次数
            print('[running] train epoch %d .' % epoch)
            n_Step = 0 #迭代数据数
            while True:
                try:
                    print('[message] epoch %d . Have train datas %d+'%(epoch, n_Step*save_Step))
                    self._model.fit_generator(yield_Datas, save_Step)
                    n_Step += 1
                except StopIteration:
                    print('[error] generator error. Please check data format.')
                    break
                self.Save_model(filepath = abspath + 'acoustic_model/' + model_Name + now_Time + '/',comment = 'e_' + str(epoch) + '_steo_' + str(n_Step*save_Step))
                self.Test_model(self.datapath, str_Data = 'train', data_Count = 100)
                self.Test_model(self.datapath, str_Data = 'cv', data_Count = 100)

    def Save_model(self, filepath = abspath + 'acoustic_model/' + model_Name , comment = ''):
        '''
        保存模型参数
        '''
        if(not os.path.exists(filepath)):
            os.makedirs(filepath)
        self._model.save_weights(filepath + comment +'.model')
        self.base_model.save_weights(filepath + comment + '.model.base')
        f = open('step' + model_Name + '.txt', 'w',encoding = 'utf-8')
        f.write(filepath + comment)
        f.close()

    def Test_model(self, datapath = '', str_Data = 'cv', data_Count = 100, out_Report = True, show_Ratio = True, io_Step_Print = 10, io_Step_File = 10):
        '''
        测试检验模型效果
	io_step_print
            为了减少测试时标准输出的io开销，可以通过调整这个参数来实现
	io_step_file
            为了减少测试时文件读写的io开销，可以通过调整这个参数来实现
	'''
        data = Acoustic_data(self.datapath , str_Data)
        num_Data = data.Get_data_num() # 获取数据的数量
        if(data_Count <= 0 or data_Count > num_Data): # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_Count = num_Data
        try:
            random_Num = random.randint(0,num_Data - 1) # 获取一个随机数
            words_Num = 0
            word_Error_Num = 0
            nowtime = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
            if(out_Report == True):
                if not os.path.exists('./doc/Test_Report_' + str_Data):
                    os.mkdir('./doc/Test_Report_' + str_Data)
                f = open('./doc/Test_Report_' + str_Data + '/' + nowtime + '.txt', 'w', encoding='UTF-8') # 打开文件并读入
            tmp = '测试报告\n模型编号 ' + model_Name + '\n\n'
            for i in range(data_Count):
                data_Input, data_Labels = data.Get_data((random_Num + i) % num_Data)  # 从随机数开始连续向后取一定数量数据
                #=data_Input, data_Labels = data.Get_data(1 , data_Count)  # 从随机数开始连续向后取一定数量数据
                # 数据格式出错处理 开始
                # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
                num_Bias = 0
                while(data_Input.shape[0] > self.AUDIO_LENGTH):
                    print('*[Error]','wave data lenghth of num',(random_Num + i) % num_Data, 'is too long.','\n A Exception raise when test Speech Model.')
                    num_Bias += 1
                    data_Input, data_Labels = data.Get_data((random_Num + i + num_Bias) % num_Data)  # 从随机数开始连续向后取一定数量数据
                    #=data_Input, data_Labels = data.Get_data(1, data_Count)  # 从随机数开始连续向后取一定数量数据
                pre = self.Predict(data_Input, data_Input.shape[0] // 8)
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
                    tmp += 'Pred:\t' + str(pre) + '\n'
                    tmp += '\n'
                print('*[Test Result] Speech Recognition ' + str_Data + ' set word error ratio: ', word_Error_Num / words_Num * 100, '%')
            if(out_Report == True):
                tmp += '*[测试结果] 语音识别 ' + str_Data + ' 集语音单字错误率： ' + str(word_Error_Num / words_Num * 100) + ' %'
                f.write(tmp)
                tmp = ''
                f.close()
        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

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
        #=print(r1)   #测试输出
        #=print('\n') #测试输出
        return r1
        pass

    def Get_edit_distance(self, str1, str2):
        leven_cost = 0
        s = difflib.SequenceMatcher(None, str1, str2)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            #print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
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
