#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import math

'''
    这个代码，是先把图片都旋转正，然后再统一旋转，保证样本均衡
'''

def rotate_to_0(lines):
    lines_new = []
    for line in lines:
        file, label = line.split(" ")
        label = label.replace("\n","")
        path, name = os.path.splitext(file)
        img = cv2.imread(file)

        if label == "0":
            cv2.imwrite(os.path.join(path + ".jpg"), img)
            label = "0"
            line_new_0 = path + ".jpg" + " " + label
            lines_new.append(line_new_0)

        if label == "1":
            rotated_1 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(path + ".jpg"),rotated_1)
            label = "0"
            line_new_1 = path + ".jpg" + " " + label
            lines_new.append(line_new_1)

        if label == "2":
            rotated_2 = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(os.path.join(path + ".jpg"), rotated_2)
            label = "0"
            line_new_2 = path + ".jpg" + " " + label
            lines_new.append(line_new_2)

        if label == "3":
            rotated_3 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(path + ".jpg"), rotated_3)
            label = "0"
            line_new_3 = path + ".jpg" + " " + label
            lines_new.append(line_new_3)
    return lines_new


def main(txt):
    lines = []
    with open(txt, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            lines.append(line)
    lines_new = rotate_to_0(lines) # 旋转正
    return lines_new

def all_rotate(lines_new):
    lines = []
    for line in lines_new:
        file, label = line.split(" ")
        label = label.replace("\n", "")
        img = cv2.imread(file)
        path, name = os.path.splitext(file)

        line_new_0 = path + ".jpg" + " " + label
        lines.append(line_new_0)

        rotated_1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.path.join(path + "_" + "1" + ".jpg"), rotated_1)
        label = "1"
        line_new_1 = path + "_" + "1" + ".jpg" + " " + label
        lines.append(line_new_1)

        rotated_2 = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(os.path.join(path + "_" + "2" + ".jpg"), rotated_2)
        label = "2"
        line_new_2 = path + "_" + "2" + ".jpg" + " " + label
        lines.append(line_new_2)

        rotated_3 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(path + "_" + "3" + ".jpg"), rotated_3)
        label = "3"
        line_new_3 = path + "_" + "3" + ".jpg" + " " + label
        lines.append(line_new_3)

    with open("data/validate_2000.txt","w",encoding='utf-8') as f1:
        for label in lines:
            f1.write(str(label) + "\n")


if __name__ == '__main__':
    txt_path = "data/validate.txt"
    # 全部旋转正
    lines_new = main(txt_path)
    # 统一旋转，样本均衡
    all_rotate(lines_new)

# 测试
# if __name__ == '__main__':
#     img = cv2.imread("data/validate/ocr_o_0pdB35ay1571813607039_pOLgTtKM1571813621891_5607930943762045072_3.jpg")
#     cv2.imshow("img",img)
#     rotate_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # 顺时针90度
#     cv2.imshow("rotate_image", rotate_image)
#     rotate1 = rotate(img, 90,scale=1.0) # 顺时针90度
#     cv2.imshow("rotate1", rotate1)
#     rotate2 = rotate(img, -90, scale=1.0) # 逆时针90度
#     cv2.imshow("rotate2", rotate2)
# cv2.waitKey()
