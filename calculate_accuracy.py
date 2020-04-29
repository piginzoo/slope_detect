#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
    这个代码用来帮我们计算三个旋转模型的正确率比较：
    - 老模型（大图）--2019-05-07
    - 赵毅模型
    - 新模型（切图）--2020-04-17
'''

def read_file(path):
    lines = open(path, "r", encoding='utf-8')
    test_files = []
    test_labels = []
    for line in lines:
        file, label = line.split(" ")
        #print("修改之前file:",file)
        file = file[:-4] + ".jpg"
        #print("修改之后file:",file)
        label = label.replace("\n", "")
        test_files.append(file)
        test_labels.append(label)
    return test_files, test_labels

def compare(test_files, test_labels, path):
    print("=================================================================================")
    print("                                                                    标签   |  识别")
    print("=================================================================================")
    correct = 0
    lines = open(path, "r", encoding='utf-8')
    for line in lines:
        name, label = line.split(" ")
        name = name[:-4] + ".jpg"
        label = label.replace("\n", "")

        l = test_labels[test_files.index(name)]
        if l == label:
            correct += 1
        else:
            print("[", name, "]不一致：", label, " vs ", l)
    return correct


def calculate_accuracy(i,j,k):
    print("--------统计2000张图片预测正确的个数--------")
    print("老模型预测正确的个数：", i)
    print("新模型预测正确的个数：", j)
    print("赵毅模型预测正确的个数：", k)
    print("-------------比较模型的正确率-------------")
    print("老模型的正确率accuracy:",   i / 2000)
    print("新模型的正确率accuracy:",   j / 2000)
    print("赵毅模型的正确率accuracy:", k / 2000)



if __name__ == '__main__':
    # 2000张
    validate_path = "data/pred/2000/validate_2000.txt"
    old_path = "data/pred/2000/pred_20190507.txt"
    new_path = "data/pred/2000/pred_20200428.txt"
    zhao_path = "data/pred/2000/pred_20200424_zhao.txt"


    # 测试
    # validate_path = "data/test/00.txt"
    # old_path = "data/test/11.txt"
    # new_path = "data/test/22.txt"
    # zhao_path = "data/test/33.txt"

    test_files_old, test_labels_old = read_file(old_path)
    correct_old = compare(test_files_old, test_labels_old, validate_path)

    test_files_new, test_labels_new = read_file(new_path)
    correct_new = compare(test_files_new, test_labels_new, validate_path)

    test_files_zhao, test_labels_zhao = read_file(zhao_path)
    correct_zhao = compare(test_files_zhao, test_labels_zhao, validate_path)


    calculate_accuracy(correct_old, correct_new, correct_zhao)