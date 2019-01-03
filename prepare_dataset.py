#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified: 2019-01-03 10:41
# Email        : coffrelv@163.com
# Filename     : prepare_dataset.py
# Description  : 通过配置文件配置文件路径读取数据相关信息
# ******************************************************

import sys
import os
import re
import shutil
import codecs

import read_config

#将所有训练所需语料相关数据收集并存储
def Get_all_text_data(Sign):
    Wav_path = read_config.Read_config('doc/config.cfg', 'section1', Sign+'_wav_path')
    f = open('doc/'+Sign+'all_text.txt' , mode = 'w' , encoding = 'utf-8')
    Filenamelist = []
    for parent, dirname , filenames in os.walk(Wav_path, topdown = True):
        for filename in filenames:
            if not filename.endswith('.wav'):
                continue
            else:
                file_path = os.path.join(parent, filename)
                Filenamelist.append(file_path)
        Filenamelist.sort()
        num = 1
        for i in Filenamelist:
            f_txt = open(i[:-4]+'.txt', mode = 'r', encoding = 'utf-8')
            wav_txt = f_txt.readline()
            wav_txt = re.sub('[^a-zA-Z ]','',wav_txt)
            wav_txt = wav_txt.strip()
            f.write(i[-13:-4]+'_'+i[-22:-14]+' '+i+' '+wav_txt+"\n")
            num += 1
    f.close()

#本脚本主函数
def Main_self(Train_signal,Test_signal, dev_Signal):
    Get_all_text_data(Train_signal)
    Get_all_text_data(Test_signal)
    Get_all_text_data(dev_Signal)

if __name__ == '__main__':
    Train_signal = 'train'
    Test_signal = 'test'
    dev_Signal = 'cv'
    Main_self(Train_signal, Test_signal, dev_Signal)
    exit()
