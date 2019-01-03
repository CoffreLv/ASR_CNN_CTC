#!/usr/bin/python 
# ******************************************************
# Author       : CoffreLv
# Last modified: 2019-01-03 10:30
# Email        : coffrelv@163.com
# Filename     : read_config.py
# Description  : 读写配置文件
# ******************************************************

import sys
import os
import configparser

#定义配置文件路径为全局变量
Configfile_path = 'doc/config.cfg'

#将配置写入路径为‘config_path’的配置文件
def Write_config(config_path):
    config = configparser.RawConfigParser()
    section_ch = input("请输入你要配置的Section：")
    config.add_section(section_ch)
    while 1:
        choice = input("如果要添加配置请输入“y”否则请输入“n”：")
        if choice == "y":
            config_name = input("请输入你要配置的配置名称：")
            config_value = input("请输入你要配置的配置值：")
            config.set(section_ch, config_name, config_value)
        elif choice == "n":
            break
        else:
            print("输入错误！请重新输入！")
            continue
    with open(config_path , 'w') as configfile:
        config.write(configfile)

#读取路径为‘config_path’的配置文件中的‘section_ch’组中‘congfig_key’的配置值打印并返回
def Read_config(config_path, section_ch, config_key):
    config = configparser.RawConfigParser()
    config.read(config_path)
    config_dict = {}
    config_ch = config[section_ch][config_key]
    print(config_key+"\t"+config_ch)
    return config_ch

#读取路径为‘config_path’的配置文件中的‘section_ch’组的全部配置并打印
def Read_all_config(config_path, section_ch):
    config = configparser.RawConfigParser()
    config.read(config_path)
    for key in config[section_ch]:
        print(key+'\t'+config[section_ch][key])

#本脚本主函数
def Main_self():
    while 1:
        choice = input("写配置请输入“1”，读配置请输入“2”，退出请输入“0”：")
        if choice == '1':
            Write_config(Configfile_path)
        elif choice == '2':
            section_choice = input("请输入选择的section：")
            Read_all_config(Configfile_path, section_choice)
        elif choice == '0':
            break
        else:
            print("输入错误！请重新输入！")

if __name__ == '__main__':
    Main_self()
    exit()
