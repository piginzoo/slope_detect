#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image

# POSSIBILITY_SPECIAL_1 = 0.1
# POSSIBILITY_SPECIAL_2 = 0.3
# POSSIBILITY_SPECIAL_3 = 0.2

# # 随机接受概率
# def _random_accept(accept_possibility):
#     return np.random.choice([True,False], p = [accept_possibility,1 - accept_possibility])

def random_rotate(lines):
    # if not _random_accept(0.2): return lines,s #不旋转
    lines_new = []
    i = 1

    for line in lines:
        file, label = line.split(" ")
        label = label.replace("\n","")

        if label == "0":
            img = Image.open(file)
            line_new_1 = rotate_270(file,img)
            line_new_2 = rotate_180(file,img)
            line_new_3 = rotate_90(file,img)
            lines_new.append(line_new_1)
            lines_new.append(line_new_2)
            lines_new.append(line_new_3)
            #print("lines_new:",lines_new)
            i += 1
            if i > 1200:
                break
        else:
            continue

    print("旋转了图片[%s]张:",i)
    return lines_new


# 指定逆时针旋转的角度
def rotate_270(file,img):
    #img = Image.open(file)
    img_rotate = img.rotate(270)
    path, name = os.path.splitext(file)
    img_rotate.save(os.path.join(path + "_" + "1" + ".jpg"))
    label = "1"
    line_new = path + "_" + "1" + ".jpg" + " " + label
    return line_new

def rotate_180(file,img):
    #img = Image.open(file)
    img_rotate = img.rotate(180)
    path, name = os.path.splitext(file)
    img_rotate.save(os.path.join(path + "_" + "2" + ".jpg"))
    label = "2"
    line_new = path + "_" + "2" + ".jpg" + " " + label
    return line_new

def rotate_90(file,img):
    #img = Image.open(file)
    img_rotate = img.rotate(90)
    path, name = os.path.splitext(file)
    img_rotate.save(os.path.join(path + "_" + "3" + ".jpg"))
    label = "3"
    line_new = path + "_" + "3" + ".jpg" + " " + label
    return line_new

def gray(file,label,image):
    #image = cv2.imread(file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    path, name = os.path.splitext(file)
    cv2.imwrite(os.path.join(path + "_" + "gray" + ".jpg"),image_gray)
    line_new = path + "_" + "gray" + ".jpg"+ " " + label
    return line_new

def noise(file,label,img):
    #img = cv2.imread(file)
    for i in range(20): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = np.random.randint(255)
    path, name = os.path.splitext(file)
    cv2.imwrite(os.path.join(path + "_" + "noise" + ".jpg"), img)
    line_new = path + "_" + "noise" + ".jpg" + " " + label
    return line_new


def main(txt):
    lines = []
    lines_enhance = []
    with open(txt, "r", encoding='utf-8') as f:
        for line in f.readlines():
            lines.append(line)

    lines_new = random_rotate(lines) # 旋转

    # 增强
    for line in lines_new:
        file, label = line.split(" ")
        label = label.replace("\n", "")
        image = cv2.imread(file)
        line_gray = gray(file, label,image)
        line_noise = noise(file, label,image)
        lines_enhance.append(line_gray)
        lines_enhance.append(line_noise)
    lines_all = lines + lines_new + lines_enhance

    with open("data/train_new.txt", "w", encoding='utf-8') as f1:
        for line in lines_all:
           f1.write(str(line) + "\n")


if __name__ == '__main__':
    #img_dir = "data/train"
    txt = "data/train.txt"
    main(txt)










