#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

'''
    这个代码用来帮我们计算三个旋转模型的正确率比较：
    - 老模型（大图）--2019-05-07
    - 赵毅模型
    - 新模型（切分小图）--2020-04-07
'''
#
# validate_path = "data/pred/validate.txt"
# old_path = "data/pred/pred_20190507.txt"
# new_path = "data/pred/pred_20200407.txt"
#
#
# labels_old = []
# with open(old_path, "r", encoding='utf-8') as f:
#     for line in f.readlines():
#         #print('line:', line)
#         name,label = line.split(" ")
#         label = label.replace("\n", "")
#         labels_old.append(label)
#         #print('labels_old:', labels_old)
#
#
# labels_new = []
# with open(new_path, "r", encoding='utf-8') as f:
#     for line in f.readlines():
#         #print('line:', line)
#         name,label = line.split(" ")
#         label = label.replace("\n", "")
#         labels_new.append(label)
#         #print('labels_new:', labels_new)
#
#
# lines = []
# with open(validate_path, "r", encoding='utf-8') as f:
#     for line in f.readlines():
#         #print('line:', line)
#         line = line.replace("\n", "")
#         lines.append(line)
#         #print('lines:', lines)
#
# i = 0
# all_txt = []
# while i <= 499:
#     txt = lines[i] + " " + labels_old[i] + " " + labels_new[i]
#     i +=1
#     all_txt.append(txt)
#     continue
#
#
# with open("data/pred/all.txt", "w", encoding="utf-8") as f:
#     for txt in all_txt:
#         f.write(txt +"\n")

i = 0
j = 0
m = 0
n = 0
a = 0
b = 0
l = 0
with open("data/pred/all.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        name,label,label1,label2 = line.split(" ")
        label2 = label2.replace("\n", "")

        print("-----------统计预测正确的个数-------------")
        if label == label1:
            i +=1
            print("老模型预测正确的个数：", i)
        if label == label2:
            j +=1
            print("新模型预测正确的个数：", j)

        print("-----------统计原来正类的个数-------------")
        if label == "0":
            l += 1
            print("老模型正类预测为正类的个数：", l)

        print("-----------统计正类预测为正类的个数-------------")
        if label == label1 and label == "0":
            m += 1
            print("老模型正类预测为正类的个数：", m)
        if label == label2 and label == "0":
            n += 1
            print("新模型正类预测为正类的个数：", n)

        print("-----------统计所有预测为正类的个数-------------")
        if label1 == "0":
            a += 1
            print("老模型所有预测为正类的个数：", a)
        if label2 == "0":
            b += 1
            print("新模型所有预测为正类的个数：", b)

acc_old = i/500
pre_old = m / a
rec_old = m / l

acc_new = j/500
pre_new = n / b
rec_new = n / l

print("--------计算比较正确率---------") # https://blog.csdn.net/u011630575/article/details/80250177
print("老模型的正确率accuracy：", acc_old)
print("老模型的精确率precision：", pre_old)
print("老模型的召回率recall：", rec_old)
print("老模型的F1：", pre_old * rec_old * 2 /(pre_old + rec_old))

print("新模型的正确率accuracy：", acc_new)
print("新模型的精确率precision：", pre_new)
print("新模型的召回率recall：", rec_new)
print("新模型的F1：", pre_new * rec_new * 2 /(pre_new + rec_new))




