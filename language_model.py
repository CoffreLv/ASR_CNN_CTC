#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified:	2019-03-12 08:24
# Email        : coffrelv@163.com
# Filename     :	language_model.py
# Description  :    从柠檬博客ＡＳＲＴ借鉴过来的语言模型,
#                   尚未启用 
# ******************************************************

import platform as plat

class language_Model(): #语言模型类
    def __init__(self, model_Path):
        self.model_Path = model_Path
        self.slash = '/'
        if (self.slash != self.model_Path[-1]):
            self.model_Path = self.model_Path + self.slash

        pass

    def load_Model(self):
        self.dict_Mark = self.get_Symbol_Dict('lexicon.txt')
        self.model1 = self.get_Language_Model(self.model_Path + 'language_model1.txt')
        self.model2 = self.get_Language_Model(self.model_Path + 'language_model2.txt')
        self.mark = self.get_Mark(self.model_Path + 'dict_mark')
        model = (self.dict_Mark, self.model1, self.model2)
        return model
        pass

    def speech_To_Text(self, list_Syllable):
        '''
        为语音识别专用的处理函数
        实现从语音拼音符号到最终文本的转换
        '''

        r = ''
        length = len(list_Syllable)
        if(length == 0):    #传入的参数没有包含任何标注
            return ''
        str_Tmp = [list_Syllable[0]]    #先取出标注列表中的第一个字
        for i in range(0, length -1):
            str_Split = list_Syllable[i] +' ' + list_Syllable[i+1]  #依次从第一个字开始每次连续取两个字标注
            if(str_Split in self.mark): #如果这个标注在标注状态转移字典中
                str_Tmp.append(list_Syllable[i+1])  #将第二个字的标注加入
            else:
                str_Decode = self.decode(str_Tmp, 0.0000)   #否则不加入，然后将现有的拼音序列进行解码
                if(str_Decode != []):
                    r += str_Decode[0][0]
                str_Tmp = [list_Syllable[i+1]]  #再次从i+1开始作为第一个拼音

        str_Decode = self.decode(str_Tmp, 0.0000)
        if(str_Decode != []):
            r += str_Decode[0][0]
        return r

    def decode(self, list_Syllable, pre_Set = 0.0001):
        #实现标注到文本的转换，基于马尔科夫链
        list_Words = []

        num_Mark = len(list_Syllable)
        #开始语音解码
        for i in range(num_Mark):
            ls = ''
            if(list_Syllable[i] in self.dict_Mark): #如果标注存在于标注字典中
                ls = self.dict_Mark[list_Syllable[i]]   #获取标注对应的字，ls包含了该标注对应的所有字
            else:
                break

            if(i == 0): #对第一个字做初试处理
                num_ls = len(ls)
                for j in range(num_ls):
                    tuple_Word = ['', 0.0]  #设置马尔科夫模型初始状态值
                    tuple_Word = [ls[j], 1.0]   #设置初始概率，设为1.0
                    list_Words.append(tuple_Word)   #添加到可能的句子列表

                continue
            else:
                list_Words_2 = []   #开始处理紧跟在第一个字后面的字
                num_ls_Word = len(list_Words)
                for j in range(0, num_ls_Word):
                    num_ls = len(ls)
                    for k in range(0, num_ls):
                        tuple_Word = ['', 0.0]
                        tuple_Word = list[list_Words[j]]    #把现有的每一条短语读取出来
                        tuple_Word[0] = tuple_Word[0] + ls[k]   #尝试按照下一个音可能对应的全部的字进行组合
                        tmp_Words = tuple_Word[0][-2:]  #取出用于计算的最后两个字
                        if(tmp_Words in self.model2):   #判断它们是不是在状态转移表里
                            tuple_Word[1]  = tuple_Word[1] * float(self.model2[tmp_Words] / float(self.model1[tmp_Words[-2]])) #核心：当前概率上乘转移概率，公式化简后维第n+1和n个字出现的次数除以第n-1个字出现的次数
                        else:
                            tuple_Word[1] = 0.0
                            continue
                        if(tuple_Word[1] >= pow(pre_Set, i)):   #大于阈值之后保留，否则丢弃
                            list_Words_2.append(tuple_Word)
                list_Words = list_Words_2

        for i in range(0, len(list_Words)):
            for j in range(i+1, len(list_Words)):
                if(list_Words[i][1] < list_Words[j][1]):
                    tmp = list_Words[i]
                    list_Words[i] = list_Words[j]
                    list_Words[j] = tmp

        return list_Words
        pass

    def get_Symbol_Dict(self, dict_Filename):
        #读取标注字典文件，返回读取后的字典
        txt_OBJ = open(dict_Filename, 'r', encoding = 'UTF-8')
        txt_Text = txt_OBJ.read()
        txt_OBJ.close()
        txt_Lines = txt_Text.split('\n')    #按行分割

        dict_Symbol = {}    #初始标注字典
        for i in txt_Lines:
            list_Symbol = []    #初始化标注列表
            if(i != ''):
                txt_L = i.split('\t')
                mark = txt_L[0]
                for word in txt_L[1]:
                    list_Symbol.append(word)
            dict_Symbol[mark] = list_Symbol
        return dict_Symbol

    def get_Language_Model(self, model_Language_Filename):
        #读取语言模型文件，返回读取后的模型
        txt_OBJ = open(model_Language_Filename, 'r', encoding = 'UTF-8')
        txt_Text = txt_OBJ.read()
        txt_OBJ.close()
        txt_Lines = txt_Text.split('\n')

        dict_Model = {}
        for i in txt_Lines:
            if (i != ''):
                txt_L = i.split('\t')
                if(len(txt_L) == 1):
                    continue
                dict_Model[txt_L[0]] = txt_L[1]

        return dict_Model

    def get_Mark(self, filename):
        file_OBJ = open(filename, 'r', encoding = 'UTF-8')
        txt_All = file_OBJ.read()
        file_OBJ.close()

        txt_Lines = txt_All.split('\n')
        dict_Marks = {}
        for line in txt_Lines:
            if(line == ''):
                continue
            mark_Split = line.split('\t')
            list_Mark = mark_Split[0]
            if(list_Mark not in dict_Marks and int(mark_Split[1]) > 1):
                dict_Marks[list_Mark] = mark_Split[1]

        return dict_Marks

if(__name__ == '__main__'):
    ml = language_Model('model_language')
    ml.load_Model()
    str_Mark = ['A','vv','ta']
    r = ml.speech_To_Text(str_Mark)
    print('语音转文字结果：\n', r)
