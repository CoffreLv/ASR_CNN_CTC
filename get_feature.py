#!/usr/bin/python
# ******************************************************
# Author       : CoffreLv
# Last modified:	2018-12-18 15:44
# Email        : coffrelv@163.com
# Filename     :	get_feature.py
# Description  :
# ******************************************************

import os
import wave
import random

import numpy as np
from scipy.fftpack import fft


#汉明窗
x = np.linspace(0, 400 - 1, 400, dtype = np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi *(x) / (400-1))

class Acoustic_data():

    def __init__(self, path, type, LoadToMem = False, MemWavCount = 10000):
        '''
        初始化
        参数：
                path:数据存放位置根目录
        '''

        self.datapath = path    #数据存放位置根目录
        self.type = type    #数据类型，训练集（train）、验证集（dev）、测试集（test）
        self.slash = '/'
        if(self.slash != self.datapath[-1]):    #在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash

        self.dic_Wavlist = {}
        self.dic_Symbollist = {}

        self.symbol_Num = 0 #标注符号的数量
        self.list_Symbol = self.Get_symbol_list()   #标注符号列表
        self.list_Wav_Num = []  #wav文件标记列表
        self.list_Symbol_Num = []   #symbol标价列表

        self.data_Num = 0 #数据个数
        self.Load_data_list()

        self.wavs_data = []
        self.LoadToMem = LoadToMem
        self.MemWavCount = MemWavCount
        pass

    def Get_symbol_list(self):
        '''
        加载标注符号列表，用于标记符号
        返回一个列表list类型变量
        '''
        f = open('lexicon.txt', 'r', encoding = 'UTF-8')
        f_Text = f.read()   #读取全部字典数据
        symbol_Lines = f_Text.split('\n')   #分割字典数据
        list_Symbol = []
        for i in symbol_Lines:
            if(i!=''):
                tmp = i.split('\t')
                list_Symbol.append(tmp[0])
        f.close()
        list_Symbol.append('_')
        self.symbol_Num = len(list_Symbol)
        return list_Symbol


    def Load_data_list(self):
        '''
        加载用于计算的数据列表
        参数：
                type:选取的数据集
        '''
        if(self.type == 'train'):
            filename_Wavlist = 'text' + self.slash + 'train.wav.lst'
            filename_Symbollist = 'text' + self.slash + 'train.syllable.txt'
        elif(self.type == 'cv'):
            filename_Wavlist = 'text' + self.slash + 'cv.wav.lst'
            filename_Symbollist = 'text' + self.slash + 'cv.syllable.txt'
        elif(self.type == 'test'):
            filename_Wavlist = 'text' + self.slash + 'test.wav.lst'
            filename_Symbollist = 'text' + self.slash + 'test.syllable.txt'
        else:
            filename_Wavlist = ''   #默认为空
            filename_Symbollist = ''
        test_Path = self.datapath + filename_Wavlist
        self.dic_Wavlist,self.list_Wav_Num = Get_wav_list(test_Path)
        self.dic_Symbollist,self.list_Symbol_Num = Get_wav_symbol(self.datapath + filename_Symbollist)
        self.data_Num = self.Get_data_num()

    def Get_data_num(self):
        '''
        获取数据数量
        当wav数量和symbol数量一致时返回正确值，否则返回-1，表示出错
        '''
        num_Wavlist = len(self.dic_Wavlist)
        num_Symbollist = len(self.dic_Symbollist)
        if(num_Wavlist == num_Symbollist):
            data_Num = num_Wavlist
        else:
            data_Num = -1
        return data_Num

    def data_Genetator_All(self, batch_size = 32, audio_length = 1600):
        '''
        数据生成器函数，用于keras的generator_fit训练
        参数：
                batch_size:一次产生的数据量
                audio_length:音频长度（约16s）
        '''
        counter = 0
        batch_Size = batch_size
        labels = []
        for i in range(0,batch_size):
            labels.append([0.0])

        labels = np.array(labels, dtype = np.float)

        while counter*batch_Size < self.data_Num:
            batch_size = batch_Size
            X = np.zeros((batch_size, audio_length, 200, 1), dtype = np.float)
            y = np.zeros((batch_size, 64), dtype = np.int16)

            input_Length = []
            label_Length = []
            if (counter+1)*batch_size > self.data_Num:
                batch_size = batch_size - (counter+1) * batch_size + self.data_Num
            for i in range(batch_size):
                data_Be_Gotten_Num = counter*batch_size + i  #获取数据的编号
                data_Input , data_Labels = self.get_Data(data_Be_Gotten_Num)    #获取编号数据
                input_Length.append(data_Input.shape[0] // 8 + data_Input.shape[0] %8)
                X[i,0:len(data_Input)] = data_Input
                y[i,0:len(data_Labels)] = data_Labels
                label_Length.append([len(data_Labels)])
            counter += 1
            label_Length = np.matrix(label_Length)
            input_Length = np.array(input_Length).T
            yield [X, y, input_Length, label_Length ],labels
        pass

    def Data_genetator(self, batch_size = 32, audio_length = 1600):
        '''
        数据生成器函数，用于keras的generator_fit训练
        参数：
                batch_size:一次产生的数据量
                audio_length:音频长度（约16s）
        '''
        labels = []
        for i in range(0,batch_size):
            labels.append([0.0])

        labels = np.array(labels, dtype = np.float)

        while True:
            X = np.zeros((batch_size, audio_length, 200, 1), dtype = np.float)
            y = np.zeros((batch_size, 64), dtype = np.int16)

            input_Length = []
            label_Length = []

            for i in range(batch_size):
                random_Num = random.randint(0, self.data_Num - 1)   #获取一个随机数
                data_Input , data_Labels = self.Get_data(random_Num)    #根据随机数选取一个数据
                input_Length.append(data_Input.shape[0] // 8 + data_Input.shape[0] %8)
                X[i,0:len(data_Input)] = data_Input
                y[i,0:len(data_Labels)] = data_Labels
                label_Length.append([len(data_Labels)])

            label_Length = np.matrix(label_Length)
            input_Length = np.array(input_Length).T
            yield [X, y, input_Length, label_Length ],labels
        pass

    def get_Data(self, num_Start, num_Amount = 1):
        '''
        读取数据，返回神经网络输入和输出矩阵（可直接用于训练网络）
        参数：
                num_Start:开始选取数据的编号
                num_Amount:选取的数据数量，默认为1，即一次一个wav文件
        返回：
                三个包含wav特征矩阵的神经网络输入值，和一个标定的类别矩阵神经网络输出值
        '''
        filepath = self.dic_Wavlist[self.list_Wav_Num[num_Start]]
        list_Symbol = self.dic_Symbollist[self.list_Symbol_Num[num_Start]]
        wav_Signal, fs = Read_wav_data( filepath)

        feat_Out = []

        for i in list_Symbol:
            if (i != ''):
                tmp = self.Symbol_to_num(i)
                feat_Out.append(tmp)

        data_Input = Get_frequecy_feature(wav_Signal, fs)
        data_Input = data_Input.reshape(data_Input.shape[0], data_Input.shape[1], 1)
        data_Label = np.array(feat_Out)

        return data_Input, data_Label

    def Get_data(self, num_Start, num_Amount = 1):
        '''
        读取数据，返回神经网络输入和输出矩阵（可直接用于训练网络）
        参数：
                num_Start:开始选取数据的编号
                num_Amount:选取的数据数量，默认为1，即一次一个wav文件
        返回：
                三个包含wav特征矩阵的神经网络输入值，和一个标定的类别矩阵神经网络输出值
        '''
        ratio = 2
        if(self.type == 'train'):
            ratio = 11
        if(num_Start % ratio == 0):
            filepath = self.dic_Wavlist[self.list_Wav_Num[num_Start // ratio]]
            list_Symbol = self.dic_Symbollist[self.list_Symbol_Num[num_Start //ratio]]
        else:
            filepath = self.dic_Wavlist[self.list_Wav_Num[num_Start // ratio]]
            list_Symbol = self.dic_Symbollist[self.list_Symbol_Num[num_Start //ratio]]
        wav_Signal, fs = self.Read_wav_data( filepath)

        feat_Out = []

        for i in list_Symbol:
            if (i != ''):
                tmp = self.Symbol_to_num(i)
                feat_Out.append(tmp)

        data_Input = self.Get_frequecy_feature(wav_Signal, fs)
        data_Input = data_Input.reshape(data_Input.shape[0], data_Input.shape[1], 1)
        data_Label = np.array(feat_Out)

        return data_Input, data_Label

    def Get_symbol_num(self):
        '''
        获取符号数量
        '''
        return len(self.list_Symbol)

    def Symbol_to_num(self, symbol):
        '''
        将符号转为数字
        '''
        if(symbol != ''):
            return self.list_Symbol.index(symbol)
        return self.symbol_Num

def Get_wav_list(self,filepath):
    '''
    读取wav文件列表，返回一个存储该列表的字典
    '''
    f = open(filepath, 'r', encoding = 'UTF-8')
    f_Text = f.read()   #读取列表
    wavlist_Lines = f_Text.split('\n')   #分割列表
    dic_Wav_Filelist = {}   #初始化字典
    list_Wavmark = []   #初始化wav列表
    for i in wavlist_Lines:
        if(i != ''):
            tmp = i.split(' ')
            dic_Wav_Filelist[tmp[0]] = tmp[1]
            list_Wavmark.append(tmp[0])
    f.close()
    return dic_Wav_Filelist,list_Wavmark

def Get_wav_symbol(self,filepath):
    '''
    读取数据集中，wav文件对应的标注符号
    返回一个存储标注符号集的字典
    '''
    f = open(filepath, 'r', encoding = 'UTF-8')
    f_Text = f.read()
    symbol_Lines = f_Text.split('\n')
    dic_Symbollist = {}
    list_Symbolmark = []
    for i in symbol_Lines:
        if(i !=''):
            tmp = i.split(' ')
            dic_Symbollist[tmp[0]] = tmp[1:]
            list_Symbolmark.append(tmp[0])
    f.close()
    return dic_Symbollist,list_Symbolmark

def Read_wav_data(filepath):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    wav = wave.open(filepath, 'rb') #打开wav音频文件流
    num_Frame = wav.getnframes()    #获取帧数
    num_Channel = wav.getnchannels()    #获取声道数
    frame_Rate = wav.getframerate() #获取帧速率
    num_Sample_Width = wav.getsampwidth()   #获取比特宽度，即每一帧的字节数
    str_Data = wav.readframes(num_Frame)    #获取全部帧数据
    wav.close() #关闭音频流
    wave_Data = np.fromstring(str_Data, dtype = np.short)   #将声音文件转换为np矩阵
    wave_Data.shape = -1, num_Channel   #按照声道数将数组变维，单声道一列，双声道两列
    wave_Data = wave_Data.T #转置矩阵
    return wave_Data, frame_Rate

def Get_frequecy_feature(wav_Signal, fs):
    '''
    对输入数据进行处理,加窗？？？？？？
    '''
    time_Window = 25   #单位ms
    window_Length = fs / 1000 * time_Window #计算窗长的公式，目前全部为400固定值

    wav_Array = np.array(wav_Signal)
    wav_Length = wav_Array.shape[1]

    range_End = int(len(wav_Signal[0]) / fs * 1000 - time_Window) // 10 #计算循环终止的位置，也就是最终生成的窗数
    data_Input = np.zeros((range_End, 200), dtype = np.float)   #用于存放最终的频率特征数据
    data_Line = np.zeros((1, 400), dtype = np.float)

    for i in range(0, range_End):
        p_Start = i * 160
        p_End = p_Start + 400
        data_Line = wav_Array[0, p_Start:p_End]
        data_Line = data_Line * w   #加汉明窗
        data_Line = np.abs(fft(data_Line)) / wav_Length
        data_Input[i] = data_Line[0: 200]   #设置维400除以2是取一半数据，因为是对称的
    data_Input = np.log(data_Input +1)
    return data_Input

