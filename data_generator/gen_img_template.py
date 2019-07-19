# -*- coding: utf-8 -*-

import datetime
import os
import random

import cv2 as cv
import keras
import numpy as np
import tensorflow as tf
from PIL import Image

print("tensorflow", tf.__version__)
print("keras", keras.__version__)
print("cv", cv.__version__)
print("np", np.__version__)

# 常量
global ROOT
global BACKGB
global IMAGES
global OUT
global all_bg_images


def filter_file_subfix(fileSubfix):
    if fileSubfix == ".jpg" or fileSubfix == ".jpeg" or fileSubfix == ".JPG" or fileSubfix == ".JPEG" or fileSubfix == ".png" or fileSubfix == ".PNG":
        return True
    else:
        return False


# load all bg img
def load_all_backgroud_images(bground_path):
    bground_list = []
    for img_name in os.listdir(bground_path):
        (filepath, fileName) = os.path.split(img_name)
        name, subfix = os.path.splitext(fileName)
        if filter_file_subfix(subfix):
            image = Image.open(bground_path + img_name)
            image = image.convert("RGBA")
            bground_list.append(image)
            print("    加载背景图片：", bground_path + img_name)

    print("背景总共数量", len(bground_list))
    return bground_list


# 角度集合
angleList = {
    0: 0,
    1: 90,
    2: 180,
    3: 270
}


# 按分类获取一个随机角度
def get_angle(idx):
    al = angleList[idx]
    angle = random.randint(-5, 5)
    angle = angle + al
    return angle


# print(random_angle(1)),exit(0)

# 获取一个随机的宽高
def get_random_wh():
    w = random.randint(20, 100)
    h = random.randint(20, 100)
    return w, h


# 获取一个仿射数值
def get_random_affine_offset(angle, rw, rh):
    """
    :return 类型 0上 1左 2下 3右 ,x,y
    """
    direction = 0
    if angle <= 1:
        direction = random.randint(0, 1)
    elif angle > 1 and angle <= 3:
        direction = random.randint(2, 3)

    x = random.randint(0, rw)
    y = random.randint(0, rh)
    return direction, x, y


# print(get_random_affine_offset(30, 55)), exit(0)


# 背影图和主图合并 生成正常图片
def gen_new_normal_img(bgimg, mainimg):
    rx, ry = get_random_wh()
    mainimg = mainimg.convert("RGBA")
    mw, mh = mainimg.size
    print("主图:", mw, mh)
    # 让背景图宽高大于主图，重置背景图宽高
    bgw = mw + (rx * 2)
    bgh = mh + (ry * 2)
    # 重置背景图
    newBgImg = bgimg.resize((bgw, bgh), Image.ANTIALIAS)
    # 合并图片
    newBgImg.paste(mainimg, (rx, ry), mainimg)
    # 保存合并后的图片
    newBgImg.save(IMAGES + "/newimage_F" + ".png")


# 根据主图宽高类型，随机获取一个背景图，直到获取同样类型的背景图为址
# 最多尝试获取次数为 c < 10
def get_random_bg_img(mainimg, c):
    """
    根据主图类型获取背景图
    :param mainimg:主图像
    :return:
    """
    # print("获取背景图次数", c)
    # 0-宽图 1-高图
    mainimgType = 1
    bgimgType = 1
    mw, mh = mainimg.size
    if mw >= mh:
        mainimgType = 0

    # 随机背景图
    ranBgimg = random.choice(all_bg_images)
    # ranBgimg = Image.open(BACKGB + fileName).convert("RGBA")
    bgw, bgh = ranBgimg.size
    # print(bgw, bgh)
    if bgw >= bgh:
        bgimgType = 0

    # 递归获取了10次直接返回
    if c > 10:
        print("获取背景图次数大于10次直接返回", mainimgType, bgimgType)
        return ranBgimg

    # 判断两个图的类型是否一样，如果一样则返回
    if mainimgType == bgimgType:
        return ranBgimg
    else:
        # 两个图类型不一样，重新获取背景图
        return get_random_bg_img(mainimg, c + 1)


# print(get_random_bg_img(Image.open("images/20190402051418692hjhjiuu.jpg"), 0)), exit(0)


def gen_rotate_img(mainimg, fileName, label_file):
    for idx in range(0, len(angleList)):  # range(0, len(angleList)):
        angle = get_angle(idx)
        startTime = datetime.datetime.now()
        rotateImg = mainimg.rotate(angle, expand=True)
        mw, mh = rotateImg.size
        print("旋转主图:", mw, mh, angle, "用时:", (datetime.datetime.now() - startTime).seconds)
        # 随机加载一个背景图
        startTime = datetime.datetime.now()
        ranBgimg = get_random_bg_img(rotateImg, 0)
        print("随机加载一个背景图用时:", (datetime.datetime.now() - startTime).seconds)
        # ranBgimg = Image.open("backgb/WechatIMG27033.jpeg").convert("RGBA")
        # 生成仿射图片
        gen_affine_img(ranBgimg, rotateImg, idx, label_file, fileName)


# 生成仿射图片
def gen_affine_img(bgimg, mainimg, angle_type, label_file, fileName=None):
    """
    1.将角度图片生成仿射图片并做背景透明处理
    2.将生成好的仿射图片与背景图合并

    :param bgimg: 背景图
    :param mainimg:主图
    :param angle_type:角度类型 上0 左1 下2 右3
    :return:
    """
    startTime = datetime.datetime.now()
    # 将Image 转换为 opencv (cv2)
    bgimg = cv.cvtColor(np.asarray(bgimg), cv.COLOR_BGRA2RGBA)
    mainimg = cv.cvtColor(np.asarray(mainimg), cv.COLOR_BGRA2RGBA)
    print("Image to CV用时:", (datetime.datetime.now() - startTime).seconds)

    startTime = datetime.datetime.now()
    # 获取一个随机的xy
    rw, rh = get_random_wh()

    # 得到主图宽高
    h, w = mainimg.shape[:2]

    # 根据角度 随机获取调整仿射
    d, x, y = get_random_affine_offset(angle_type, rw, rh)  # 类型 0上 1左 2下 3右 , x, y

    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if d == 0:
        pts2 = np.float32([[0 + x, 0], [w - y, 0], [0, h], [w, h]])
    elif d == 1:
        pts2 = np.float32([[0, 0], [w, 0], [0 + x, h], [w - y, h]])
    elif d == 2:
        pts2 = np.float32([[0, 0 + y], [w, 0], [0, h - y], [w, h]])
    elif d == 3:
        pts2 = np.float32([[0, 0], [w, 0 + y], [0, h], [w, h - y]])

    # 计算出M
    M = cv.getPerspectiveTransform(pts1, pts2)

    # 第三个参数：变换后的图像大小
    mainimg = cv.warpPerspective(mainimg, M, (w, h))
    print("调整仿射用时:", (datetime.datetime.now() - startTime).seconds)

    startTime = datetime.datetime.now()
    # 将背景图大小按主图比例重置，宽度和高度随机大于主图
    bgimg = cv.resize(bgimg, (w + rw * 2, h + rh * 2))

    # 得到仿射后的图片宽高
    rows, cols, channels = mainimg.shape

    # 获取放置图片随机位置
    ranrow, ranclo = rh, rw

    print("ranrow, ranclo", ranrow, ranclo)

    # 获得roi
    roi = bgimg[ranrow:rows + ranrow, ranclo:cols + ranclo]

    # 开始将仿射后的图片背景做透明处理
    # 1 将主图转为灰度图像
    img2gray = cv.cvtColor(mainimg, cv.COLOR_BGR2GRAY)
    # 2 将图像像素值大于10替换的为255,否则是0
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    # cv.imwrite("images/ret.png", ret)
    # 3 按位与取反得到mask
    mask_inv = cv.bitwise_not(mask)
    # cv.imwrite("images/mask_inv_img.png", mask_inv)
    # 4 在背景图指定位置（roi）提取？？？
    bg_img = cv.bitwise_and(roi, roi, mask=mask_inv)
    # cv.imwrite("images/img1_bg.png", bg_img)
    # 5 从主图中提取出主图像，此时主图背景已经是透明
    fg_img = cv.bitwise_and(mainimg, mainimg, mask=mask)
    # cv.imwrite("images/img2_fg.png", fg_img)

    # 合并背景图和主图，得到合并后的结果
    dst = cv.add(bg_img, fg_img)
    # cv.imwrite("images/dst.png", dst)
    # 将合并后的结果赋值给原背景图指定位置（ranrow,ranclo）
    bgimg[ranrow:rows + ranrow, ranclo:cols + ranclo] = dst
    # 返回图片或保存图片 名称格式：文件名_角度类型_?.png

    (filepath, fileName) = os.path.split(fileName)
    name, subfix = os.path.splitext(fileName)
    outfile = OUT + name + "_" + str(angle_type) + ".png"
    print("保存图片", outfile, angle_type)
    cv.imwrite(outfile, bgimg)

    label_file.write("data/train/" + name + "_" + str(angle_type) + ".png")
    label_file.write(" ")
    label_file.write(str(angle_type))
    label_file.write("\n")
    print("合并保存图片用时:", (datetime.datetime.now() - startTime).seconds)


if __name__ == '__main__':
    """
    origin 是原始的正图，已经排除了斜的，是为了生成训练集的。
    raw.validate是原始的正图，但是为了生成验证集的。
    raw.train里面的一张图片会按照角度，旋转4次，生成4个样本，到train目录下。
    background里面放着各种的背景图，用于合成新的图片，这里面的背景是多样化的。
    做法是：
        1、从origin里面所有的文件，做 shuffle，然后取出2000张
        2、对每一张，做4、8张变形，分别对应0、90、180、270度的旋转，并且伴随着做透射，然后贴到随机的一张背景图上
        3、把生成的每一张，保存到 data/second目录下，并且将其标签写入到second.txt中："xxxxx_1.png   1”，1表示90度旋转，例如。
        4、手工把second目录下的文件拷贝到train中，切记，要提前先备份一下train，原因是有可能再生成second的文件
        5、手工合并second.txt到train.txt，使用cat train.txt second.txt > train.new.txt
        6、validate.txt 需要手动打标记
    """
    startTime = datetime.datetime.now()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")  # 数据存放目录
    parser.add_argument("--sum")  # 处理主图像的数量
    args = parser.parse_args()
    sum = int(args.sum)
    ROOT = args.dir
    BACKGB = ROOT + "background/"
    IMAGES = ROOT + "origin/"
    OUT = ROOT + "second/"
    # 预先加载所有的纸张背景
    all_bg_images = load_all_backgroud_images(BACKGB)

    # 循环1000随机加载主图片
    mainimg_list = os.listdir(IMAGES)

    temp_mainimg_list = []

    for idx in range(0, len(mainimg_list)):
        fileName = mainimg_list[idx]
        (filepath, fileName) = os.path.split(fileName)
        name, subfix = os.path.splitext(fileName)
        if filter_file_subfix(subfix):
            temp_mainimg_list.append(fileName)

    mainimg_list = temp_mainimg_list
    # print(mainimg_list)

    random.shuffle(mainimg_list)
    milen = len(mainimg_list)
    label_file_name = os.path.join(ROOT, "second.txt")
    label_file = open(label_file_name, "w")
    print("主图像总共数量", milen)
    if milen > sum:
        milen = sum

    for idx in range(0, milen):
        fileName = mainimg_list[idx]
        mainimg = Image.open(IMAGES + fileName)
        gen_rotate_img(mainimg, fileName, label_file)

    label_file.close()

    print("总共用时", (datetime.datetime.now() - startTime).seconds)
