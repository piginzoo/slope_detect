# -*- coding: utf-8 -*-


import os

import cv2
import numpy as np

from utils import data_provider,data_util


def detect(img):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)

    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)

    # 带轮廓的图片
    cv2.imwrite("contours.png", img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    # 7. 存储中间图片
    cv2.imwrite("binary.png", binary)
    cv2.imwrite("dilation.png", dilation)
    cv2.imwrite("erosion.png", erosion)
    cv2.imwrite("dilation2.png", dilation2)

    return dilation2


def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if area < 1000:
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        print("rect is: ", rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if height > width * 1.2:
            continue

        region.append(box)

    return region


def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    # 7. 存储中间图片
    # cv2.imwrite("binary.png", binary)
    # cv2.imwrite("dilation.png", dilation)
    # cv2.imwrite("erosion.png", erosion)
    # cv2.imwrite("dilation2.png", dilation2)

    return dilation2


def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if area < 1000:
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # print("rect is: ", rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if height > width * 1.2:
            continue
        print("box", box)
        region.append(box)

    return region


def detect(img):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)

    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)

    # 带轮廓的图片
    cv2.imwrite("contours.png", img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# 剪切图片
def _cut(img):
    """
    剪切图片
    :param img:
    :return:
    """
    cut_h = 448
    cut_w = 448

    height, width, _ = img.shape

    # print(height, width)
    ch = int(height / 2)
    cw = int(width / 2)
    # print("ch, cw", ch, cw)
    cth = int(cut_h / 2)
    ctw = int(cut_w / 2)
    # print("cth, ctw", cth, ctw)

    if ch - cth <= 0:
        ch = 0
        cth = height
    else:
        ch = ch - cth
        cth = cut_h

    if cw - ctw <= 0:
        cw = 0
        ctw = width
    else:
        cw = cw - ctw
        ctw = cut_w

    # print("ch, cth, cw, ctw", ch, cth, cw, ctw)

    # h:h, w:w
    # 起始坐标h:高度  起始坐标w:宽度
    crop = img[ch:cth + ch, cw:ctw + cw]
    return crop


# 按比例缩放
def zoom(img, filename=None):
    try:
        h, w, c = img.shape
        max_w, max_h = 1024, 1024
        if (1.0 * h / max_h) > (1.0 * w / max_w):
            scale = 1.0 * h / max_h
        else:
            scale = 1.0 * w / max_w

        nw = w / scale
        # nw = nw - nw % 10
        nh = h / scale
        # nh = nh - nh % 10
        newimg = cv2.resize(img, (int(nw), int(nh)), interpolation=cv2.INTER_AREA)
        newimg = _cut(newimg)
        # if not filename is None:
        #     filename = os.path.split(filename)[1]
        #     cv2.imwrite("data/pppp/" + filename, newimg)
        # (b, g, r) = cv2.split(newimg)
        # newimg = cv2.merge([r, g, b])
        print("----------------->>>>>>>>>>>> ", h, w, c, "      ", newimg.shape, filename)
        return newimg
    except BaseException as e:
        print(str(e), img)
        exit()


if __name__ == '__main__':
    label_path = "/Users/admin/local/creditease/ai/slope_detect.v1/data/validate.test.txt";
    for idx in range(0, 10000):
        image_list, image_label = data_provider.load_validate_data(label_path, 10)
        image_list = data_util.prepare4vgg(image_list)
        print(len(image_list))

    # path = "/Users/admin/local/creditease/ai/slope_detect.v1/data/validate.test/"
    # list = os.listdir(path)
    # for fn in list:
    #     fnph = path + fn
    #     print(fnph)
    #     img = cv2.imread(fnph)
    #     img = zoom(img, fn)
    #     np.array([img])
    # 读取文件
    # fn = "10028316864_C1-4_0_0.jpg"
    # img = cv2.imread(
    #     "/Users/admin/local/creditease/ai/slope_detect.v1/data/validate/" + fn)
    # print(img.shape)
    # zoom(img, fn)

    arr = [('data/validate.test/10028319288_C1-1_0_1.jpg', 0), ('data/validate.test/10028319779_C1-9_2_1.jpg', 2),
           ('data/validate.test/10028319064_C1-1_3_0.jpg', 3),
           ('data/validate.test/ocr_o_EcS9o4J51563324375393_1Fg38Sz11563324409865_1934175886340451392.JPG', 0),
           ('data/validate.test/10028318733_C1-3_0_2.jpg', 0), ('data/validate.test/10028319454_C1-15_2_0.jpg', 2),
           ('data/validate.test/10028316864_C1-4_1_2.jpg', 1), ('data/validate.test/10028318546_C1-3_0_2.jpg', 0),
           ('data/validate.test/10028318890_C1-6_3_0.jpg', 3), ('data/validate.test/10028318419_C1-1_1_1.jpg', 1),
           ('data/validate.test/10028319707_C1-2_1_1.jpg', 1), ('data/validate.test/10028319727_C1-11_1_1.jpg', 1),
           ('data/validate.test/10028316864_C1-6_0_0.jpg', 0),
           ('data/validate.test/ocr_o_DU7bk4wc1563082394679_lFdL5F081563082405274_532484287598534800.JPG', 3),
           ('data/validate.test/10028318960_C1-19_2_1.jpg', 2), ('data/validate.test/10028319779_C1-9_3_2.jpg', 3),
           ('data/validate.test/ocr_o_b4jfWyMk1563777178047_brManD0b1563777192333_6383951208534591964.jpg', 0),
           ('data/validate.test/10028318757_C1-1_2_1.jpg', 2), ('data/validate.test/10028318525_C1-1_1_1.jpg', 1),
           ('data/validate.test/10028319205_C1-8_0_1.jpg', 0), ('data/validate.test/10028318733_C1-3_1_0.jpg', 1),
           ('data/validate.test/10028318906_C1-3_2_0.jpg', 2), ('data/validate.test/10028319353_C1-6_0_2.jpg', 0),
           ('data/validate.test/10028319245_C1-1_3_1.jpg', 3), ('data/validate.test/10028319519_C1-6_3_0.jpg', 3),
           ('data/validate.test/10028318671_C1-30_1_1.jpg', 1),
           ('data/validate.test/ocr_o_XkcKtk811563155752115_HfkvAZ8r1563155767558_5524155614020858693.JPG', 3),
           ('data/validate.test/ocr_o_8Mt8e2BX1563322257849_CQZfYjho1563322291207_9130742639720348513.JPG', 0),
           ('data/validate.test/10028319657_C1-1_0_1.jpg', 0), ('data/validate.test/10028319353_C1-7_1_2.jpg', 1),
           ('data/validate.test/10028319513_C1-3_1_1.jpg', 1), ('data/validate.test/10028319359_C1-1_2_0.jpg', 2)]

    img = data_provider.test(arr)
