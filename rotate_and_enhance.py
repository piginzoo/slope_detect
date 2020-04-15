#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import math
from PIL import Image

# POSSIBILITY_SPECIAL_1 = 0.1
# POSSIBILITY_SPECIAL_2 = 0.3
# POSSIBILITY_SPECIAL_3 = 0.2

# # 随机接受概率
# def _random_accept(accept_possibility):
#     return np.random.choice([True,False], p = [accept_possibility,1 - accept_possibility])

def rotate(image, angle, scale=1.0):
    angle = -angle
    (h, w) = image.shape[:2]  # 2
    # if center is None: #3
    center = (w // 2, h // 2)  # 4
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 5

    # 防止旋转图像丢失
    sin = math.fabs(math.sin(math.radians(angle)))
    cos = math.fabs(math.cos(math.radians(angle)))
    h_new = int(w * sin + h * cos)
    w_new = int(h * sin + w * cos)
    M[0, 2] += (w_new - w) / 2
    M[1, 2] += (h_new - h) / 2
    # 旋转后边角填充
    # rotated = cv2.warpAffine(image, M, (w_new, h_new), borderMode=cv2.BORDER_REPLICATE)
    # 白背景填充
    rotated = cv2.warpAffine(image, M, (w_new, h_new), borderValue=(254, 254, 254))
    return rotated


def random_rotate(lines):
    # if not _random_accept(0.2): return lines,s #不旋转
    lines_new = []
    i = 1

    for line in lines:
        file, label = line.split(" ")
        label = label.replace("\n","")

        if label == "0":
            img = cv2.imread(file)
            path, name = os.path.splitext(file)

            rotated_1 = rotate(img, 90, scale=1.0)
            cv2.imwrite(os.path.join(path + "_" + "1" + ".jpg"),rotated_1)
            label = "1"
            line_new_1 = path + "_" + "1" + ".jpg" + " " + label

            rotated_2 = rotate(img, 180, scale=1.0)
            cv2.imwrite(os.path.join(path + "_" + "2" + ".jpg"), rotated_2)
            label = "2"
            line_new_2 = path + "_" + "2" + ".jpg" + " " + label

            rotated_3 = rotate(img, 270, scale=1.0)
            cv2.imwrite(os.path.join(path + "_" + "3" + ".jpg"), rotated_3)
            label = "3"
            line_new_3 = path + "_" + "3" + ".jpg" + " " + label

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


def gray(file,label,image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    path, name = os.path.splitext(file)
    cv2.imwrite(os.path.join(path + "_" + "gray" + ".jpg"),image_gray)
    line_new = path + "_" + "gray" + ".jpg"+ " " + label
    return line_new

def noise(file,label,img):
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
            line = line.replace("\n", "")
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










