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

def read_label(path):
    labels = []
    with open(path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            name, label = line.split(" ")
            label = label.replace("\n", "")
            labels.append(label)
    return labels

def merge(old_path, new_path, validate_path, zhao_path):
    labels_old = read_label(old_path)
    labels_new = read_label(new_path)
    labels_zhao = read_label(zhao_path)

    lines = []
    with open(validate_path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            lines.append(line)
    return lines,labels_old,labels_new,labels_zhao


def main(lines, labels_old, labels_new,labels_zhao, merge_path):
    i = 0
    all_txt = []
    while i <= 499:
        txt = lines[i] + " " + labels_old[i] + " " + labels_new[i] + " " + labels_zhao[i]
        i +=1
        all_txt.append(txt)
        continue

    with open(merge_path, "w", encoding="utf-8") as f:
        for txt in all_txt:
            f.write(txt +"\n")


def calculate_accurate(merge_path):
    i = 0
    j = 0
    k = 0
    with open(merge_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            name, label, label1, label2,label3 = line.split(" ")
            label3 = label3.replace("\n", "")

            if label == label1:
                i += 1
            if label == label2:
                j += 1
            if label == label3:
                k += 1

    print("-----------------统计预测正确的个数---------------")
    print("老模型预测正确的个数：", i)
    print("新模型预测正确的个数：", j)
    print("赵毅模型预测正确的个数：", k)
    print("-----------------比较模型的正确率---------------")
    print("老模型的正确率accuracy:",   i / 500)
    print("新模型的正确率accuracy:",   j / 500)
    print("赵毅模型的正确率accuracy:", k / 500)



if __name__ == '__main__':
    validate_path = "data/pred/validate.txt"
    old_path = "data/pred/pred_20190507.txt"
    new_path = "data/pred/pred_20200413.txt"
    zhao_path = "data/pred/pred_20200407.txt"
    merge_path = "data/pred/all.txt"

    # 把不同模型标签合并到一个文本文件
    lines, labels_old, labels_new,labels_zhao = merge(old_path, new_path, zhao_path, validate_path)
    main(lines, labels_old, labels_new,labels_zhao, merge_path)

    # 计算正确率
    calculate_accurate(merge_path)
