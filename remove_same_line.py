#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified: 2019-01-03 09:11
# Email        : coffrelv@163.com
# Filename     : RSL.py
# Description  : 删除文件 参数1 中的重复行 并将结果保存至 文件 参数2
# ******************************************************
import sys

oldfile = sys.argv[1]
newfile = sys.argv[2]

def Remove_Same_Line():
    f =  open(oldfile , mode = 'r', encoding = 'utf-8')
    f1 = open(newfile ,mode = 'w',encoding = 'utf-8')
    line_list = []
    for line in f:
        if line in line_list:
            continue
        else:
            line_list.append(line)
            f1.write(line)
    f.close()
    f1.close()

if __name__ =='__main__':
    Remove_Same_Line()
    exit()
