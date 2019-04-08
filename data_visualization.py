#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified: 2019-04-04 09:45
# Email        : coffrelv@163.com
# Filename     : data_visualization.py
# Description  : 
# ******************************************************

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

import get_feature
from get_feature import Acoustic_data
from acoustic_model import Acoustic_model

datapath = 'dataset'


#汉明窗
x = np.linspace(0, 400 - 1, 400, dtype = np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi *(x) / (400-1))

def Wav_signal_view( wav_Signal ):   #绘制二维矩阵可视化图像
    x = []
    y = []
    for i in range(wav_Signal.shape[1]//10):
        x.append(i)
        y.append(wav_Signal[0][i*10])
    plt.figure('wav_Signal')
    plt.plot(x,y)
    plt.savefig('./doc/view/wav_Signal.jpg')
    #plt.show()

def Data_input_view(data_Input, name):
    data_Input = data_Input.flatten()
    x = []
    y = []
    for i in range(len(data_Input)):
        x.append(i)
    plt.figure(name)
    plt.plot(x,data_Input)
    plt.savefig('./doc/view/' + name + '.jpg')
    #plt.show()

def Input_data_visualization(test_Num):
    data_Test = Acoustic_data(path = 'dataset',type = 'train')
    test_Filepath = data_Test.dic_Wavlist[data_Test.list_Wav_Num[test_Num]]
    wav_Signal, frame_Rate = get_feature.Read_wav_data(test_Filepath)
    time_Window = 25   #单位ms
    window_Length = frame_Rate / 1000 * time_Window #计算窗长的公式，目前全部为400固定值

    wav_Array = np.array(wav_Signal)
    print(wav_Array.shape)
    Wav_signal_view(wav_Array)  #数据可视化wav_Signal
    wav_Length = wav_Array.shape[1]

    range_End = int(len(wav_Signal[0]) / frame_Rate * 1000 - time_Window) // 10 #计算循环终止的位置，也就是最终生成的窗数
    data_Input = np.zeros((range_End, 200), dtype = np.float)   #用于存放最终的频率特征数据
    data_Line = np.zeros((1, 400), dtype = np.float)

    tmp_1_Input = np.zeros((range_End, 400), dtype = np.float)
    tmp_2_Input = np.zeros((range_End, 400), dtype = np.float)
    tmp_3_Input = np.zeros((range_End, 400), dtype = np.float)
    for i in range(0, range_End):
        p_Start = i * 160
        p_End = p_Start + 400
        data_Line = wav_Array[0, p_Start:p_End]
        tmp_1_Input[i] = data_Line[0:400]   #数据可视化１
        data_Line = data_Line * w   #加汉明窗
        tmp_2_Input[i] = data_Line[0:400]   #数据可视化２
        data_Line = np.abs(fft(data_Line)) / wav_Length
        tmp_3_Input[i] = data_Line[0:400]   #数据可视化３
        data_Input[i] = data_Line[0: 200]   #设置维400除以2是取一半数据，因为是对称的
    data_Input = np.log(data_Input +1)
    print(data_Input.shape)
    Data_input_view(tmp_1_Input, 'tmp_1_Input')
    #plt.show()
    Data_input_view(tmp_2_Input, 'tmp_2_Input')
    #plt.show()
    Data_input_view(tmp_3_Input, 'tmp_3_Input')
    #plt.show()
    Data_input_view(data_Input, 'data_Input')
    #plt.show()
    plt.ion()
    plt.pause(3)
    plt.close()

def Layer_output_visualization():
    model_session = Acoustic_model(datapath)
    model_session.Load_Model(filename = './acoustic_model/cnn3ctc20190328_1851/e_346.model')
    model_session.Layer_output(datapath = './dataset', str_Data = 'train', data_Count =  1)

if __name__ == '__main__':
    test_Num = 1    #用于测试的数据编号
    Input_data_visualization(test_Num)
    Layer_output_visualization()
