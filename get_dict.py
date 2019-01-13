#!/usr/bin/python
# ******************************************************
# Author       : CoffreLv
# Last modified: 2019-01-02 10:36
# Email        : coffrelv@163.com
# Filename     : get_pinyin_and_word.py
# Description  : 从 蒙古语文本+ +标注 的文档中生成类似ASR中 拼音+\t+字 的字典文档
# ******************************************************

import os
import sys
import re

filepath = sys.argv[1]
#dictpath = sys.argv[2]
dictpath = 'lexicon.txt'

#通过 音频对应文本+标注 文档生成 标注+词 的文档
def Get_pinyin_and_word(filepath, dictpath):
    f_Markfile = open(filepath , 'r', encoding = 'UTF-8')
    sentence_And_Mark_List = []
    for line in f_Markfile:
        sentence_And_Mark_List.append(line)
    f_Markfile.close()
    sentence_And_Mark_Dict = {}
    for i in sentence_And_Mark_List:
        sentence = re.sub('[a-zA-Z]','',i)
        sentence = sentence.strip()
        mark = re.sub('[^a-zA-Z ]','',i)
        mark = mark.strip()
        sentence_And_Mark_Dict[sentence] = mark
    pinyin_And_Word_Dict = {}
    f_Dictfile = open(dictpath, 'w', encoding = 'UTF-8')
    for key in sentence_And_Mark_Dict.keys():
        word_Mark_List = []
        word_List = []
        word_List = key.split(' ')
        word_Mark_List = sentence_And_Mark_Dict[key].split(' ')
        #f_Dictfile.write( key + '\t' + sentence_And_Mark_Dict[key] + '\n')
        num = 0
        for word in word_List:
            f_Dictfile.write(word_Mark_List[num] + '\t' + word + '\n')
            num += 1
    f_Dictfile.close()
    Remove_same_line(dictpath)

#删除相同的行，此处用于处理字典中重复的标注
def Remove_same_line(filename):
    f =  open(filename , mode = 'r', encoding = 'utf-8')
    line_list = []
    for line in f:
        if line in line_list:
            continue
        else:
            line_list.append(line)
    f.close()
    f = open(filename, mode = 'w', encoding = 'utf-8')
    for line in line_list:
        f.write(line)
    f.close()

if __name__ == '__main__':
    Get_pinyin_and_word(filepath, dictpath)
    exit()
