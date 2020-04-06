# -*- coding: utf-8 -*-

import cv2
import numpy as np


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
        #print("box", box)
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
        #print("缩放前：", img.shape)
        newimg = cv2.resize(img, (int(nw), int(nh)), interpolation=cv2.INTER_AREA)
        #print("缩放后：", newimg.shape)
        newimg = _cut(newimg)
        #print("剪切后：",newimg.shape)
        # if not filename is None:
        #     filename = os.path.split(filename)[1]
        #     cv2.imwrite("data/pppp/" + filename, newimg)
        # (b, g, r) = cv2.split(newimg)
        # newimg = cv2.merge([r, g, b])
        return newimg
    except BaseException as e:
        print(str(e), img)
        exit()


def cut_avg(img, sum_w, sum_h):
    print("cut")
    h, w, c = img.shape
    avg_w = int(w / sum_w)
    avg_h = int(h / sum_h)
    print("avg_w =", avg_w, " avg_h =", avg_h)
    coordinates = []
    for wi in range(0, sum_w):
        for hi in range(0, sum_h):
            print(wi, hi)


if __name__ == '__main__':
    print("main.run")
    # path = "/Users/admin/local/creditease/ai/slope_detect.v1/data/validate.test/"
    # list = os.listdir(path)
    # for fn in list:
    #     fnph = path + fn
    #     print(fnph)
    #     img = cv2.imread(fnph)
    #     img = zoom(img, fn)
    #     np.array([img])

    path = "/Users/admin/local/creditease/ai/slope_detect.v1/data/validate.test/"
    fnph = path + "10028316864_C1-4_0_0.jpg"
    img = cv2.imread(fnph)
    cut_avg(img, 3, 2)
