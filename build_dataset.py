###################################################################
# File Name: build_data_set.py
# Author: CoffreLv
# mail: coffrelv@163.com
# Created Time: 2018年07月30日 星期一 10时03分47秒
#=================================================================
#-*- coding:utf-8 -*
#本文件中包含所有与数据集构建相关的函数

import random
import re
import os
import codecs
import wave

import prepare_dataset

#根据（sign）all_text.txt获得所有text、utt2spk、wav.scp三个文档
def Get_every_text(sign):
    All_text = []
    f_all =  open('./doc/'+sign+'all_text.txt', mode = 'r', encoding = 'utf-8')
    for alltext in f_all:
        if alltext[:3]==codecs.BOM_UTF8:
            alltext = alltext[3:]
        All_text.append(alltext.split(' '))
    if not os.path.exists('./dataset/text/'):
        os.mkdir('./dataset/text/')
    with open('./dataset/text/'+sign+'.wav.lst', mode = 'w', encoding = 'utf-8') as f_wav:
        for wav_text in All_text:
            f_wav.write(' '.join(wav_text[:2])+'\n')
        f_wav.close()
    with open('./dataset/text/'+sign+'.syllable.txt', mode = 'w', encoding = 'utf-8') as f_text:
        for text in All_text:
            f_text.write(text[0]+' '+' '.join(text[2:]))
        f_text.close()

#测试数据集中音频文件格式是否复合所需格式
def Get_and_test_the_wav_params(Wav_path):
    filenames = os.listdir(Wav_path)
    for filename in filenames:
        filepath = Wav_path+filename
        if filename.endswith('.WAV'):
            f = wave.open(filepath,'rb')
            params = f.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            if not nchannels == 1:
                print('ERROR:The wavefile'+filepath+'nchannels not 1!')
                exit()
            if not framerate == 16000:
                print('ERROR:The wavefile'+filepath+'framerate not 16000!')
                exit()
        if filename.endswith('.wav'):
            f = wave.open(filepath,'rb')
            params = f.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            if not nchannels == 1:
                print('ERROR:The wavefile'+filepath+'nchannels not 1!')
                exit()
            if not framerate == 16000:
                print('ERROR:The wavefile'+filepath+'framerate not 16000!')
                exit()
    return nchannels,framerate

def Main_self():
    Get_every_text('train')
    Get_every_text('test')
    Get_every_text('cv')

if __name__ == '__main__':
    Main_self()
    exit()
