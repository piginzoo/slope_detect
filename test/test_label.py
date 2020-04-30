import numpy as np
import cv2
import math
import random

def test1():
    img_list = [100,200,300,400,500]

    a=  [1,2,3,4,1]
    b= np.array(a)
    index3 = np.where(b==1)
    label_list_rotate = []
    image_list_rotate = []
    for k in index3[0]:
        img = img_list[k]
        # img = rotate(img, 90, scale=1.0)
        k = 0
        label_list_rotate.append(k)
        image_list_rotate.append(img)


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


def test2():
    origin = cv2.imread("test/1.jpg")
    # img = cv2.imread("utils/data/ocr_o_kHqEniGs1569291613729_bKvzmhEe1569291695834_1589145655907490510.JPG")

    img = rotate(origin, -90, scale=1.0)
    cv2.imwrite("test/2.jpg",img)

    img = rotate(origin, 180, scale=1.0)
    cv2.imwrite("test/3.jpg",img)

    img = rotate(origin, 90, scale=1.0)
    cv2.imwrite("test/4.jpg",img)


def test3():
    list1 = [1,2,3,4,5,6]
    list2 = ["a","b","c","d","e","f"]
    c = list(zip(list1,list2))
    print("shuffle前：",c)
    np.random.shuffle(c)
    print("shuffle后：",c)
    val = random.sample(c,3)
    print(val)

if __name__ == '__main__':
    test3()


