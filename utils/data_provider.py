# encoding:utf-8
import logging
import os
import random
import time
import traceback

from PIL import Image
import cv2
import numpy as np
import random
import math
from utils import preprocess_utils
from utils import cut
from utils.data_util import GeneratorEnqueuer

logger = logging.getLogger("data provider")


def init_logger():
    level = logging.DEBUG

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=level,
        handlers=[logging.StreamHandler()])


# def show(img, title='无标题'):
#     """
#     本地测试时展示图片
#     :param img:
#     :param name:
#     :return:
#     """
#     import matplotlib.pyplot as plt
#     from matplotlib.font_manager import FontProperties
#     font = FontProperties(fname='/Users/yanmeima/workspace/ocr/crnn/data/data_generator/fonts/simhei.ttf')
#     plt.title(title, fontsize='large', fontweight='bold', FontProperties=font)
#     plt.imshow(img)
#     plt.show()


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


def load_data(label_file):
    f = open(label_file, 'r')
    filenames = []
    labels = []
    # 从文件中读取样本路径和标签值
    # >data/train/21.png 1
    # >data/train/22.png 0
    # >data/train/23.png 2
    for line in f:
        filename, _, label = line[:-1].partition(' ')  # partition函数只读取第一次出现的标志，分为左右两个部分,[:-1]去掉回车
        if label is None or label.strip() == "":
            logger.warning("标签数据有问题，忽略：%s", line)
            continue
        filenames.append(filename)
        label = int(label)
        labels.append(label)

    logger.info("最终样本标签数量[%d],样本图像数量[%d]", len(labels), len(filenames))
    return list(zip(filenames, labels))


# 按照FLAGS.validate_num 随机从目录中产生批量的数据，用于做验证集
def load_validate_data(validate_file, batch_num):
    logger.info("加载验证validate数据：%s，加载%d张", validate_file, batch_num)
    image_label_list = load_data(validate_file)
    val_image_names = random.sample(image_label_list, batch_num)
    image_list, label_list = val_load_batch_image_labels(val_image_names)
    # return np.array(image_list), label_list
    return image_list, label_list


# 因为验证集一张张切图预测标签后取众数，有的切图<30张，所以这个函数跟训练集的分开写了
# 加载一个批次数量的图片和标签，数量为batch数
def val_load_batch_image_labels(batch):
    for image_label_pair in batch:  # 遍历所有的图片文件
        try:
            image_file = image_label_pair[0]
            label = image_label_pair[1]
            if not os.path.exists(image_file):
                logger.warning("样本图片%s不存在", image_file)
                continue
            img = cv2.imread(image_file)

            # # TODO:将一张大图切成很多小图，直接把小图灌到模型中进行训练
            image_list = preprocess_utils.get_patches(img)
            logger.debug("将图像[%s]分成%d个patches", image_file, len(image_list))
            list = [label]
            label_list = list * len(image_list)  # 小图和标签数量一致

        except BaseException as e:
            traceback.format_exc()
            logger.error("加载一个批次图片出现异常：", str(e))

    logger.debug("加载一个批次图片,切出小图[%s]张", len(image_list))
    return image_list, label_list


# 加载一个批次数量的图片和标签，数量为batch数
def _load_batch_image_labels(batch):
    image_list_all = []
    label_list_all = []
    for image_label_pair in batch:  # 遍历所有的图片文件
        try:
            image_file = image_label_pair[0]
            _, _, name = image_file.split("/")
            label = image_label_pair[1]
            if not os.path.exists(image_file):
                logger.warning("样本图片%s不存在", image_file)
                continue
            img = cv2.imread(image_file)
            logger.debug("加载样本图片:%s", image_file)

            # # TODO:将一张大图切成很多小图，再随机抽取小图灌到模型中进行训练
            image_list = preprocess_utils.get_patches(img)
            logger.debug("将图像分成%d个patches", len(image_list))
            lab_list = [label]
            label_list = lab_list * len(image_list) # 保证同一张大图切出来的小图标签一致，小图数量和标签数量相同
            image_list_all.extend(image_list)
            label_list_all.extend(label_list)

            # check
            i = 0
            for img in image_list:
                cv2.imwrite(os.path.join("data/check/cut/" + name[:-4] + "_" + str(i) + '.jpg'), img)
                i += 1

        except BaseException as e:
            traceback.format_exc()
            logger.error("加载一个批次图片出现异常：", str(e))

    #logger.debug("加载一个批次图片标签：%s", label_list_all)
    logger.debug("加载一个批次图片,切出小图[%s]张", len(image_list_all))

    image_label_list = list(zip(image_list_all, label_list_all))
    np.random.shuffle(image_label_list)
    logger.debug("shuffle了所有的小图和标签")
    val_image_names = random.sample(image_label_list, 16)
    logger.debug("一个批次随机抽取小图的数量[%d]张，准备加载...", 16)

    image_list_sample = []
    label_list_sample = []
    for image_label_pair in val_image_names:  # 遍历所有的图片文件
        try:
            image = image_label_pair[0]
            label = image_label_pair[1]
            image_list_sample.append(image)
            label_list_sample.append(label)
        except BaseException as e:
            traceback.format_exc()
            logger.error("加载一个批次图片出现异常：", str(e))

    m = 0
    for p in image_list_sample:
        cv2.imwrite(os.path.join("data/check/sample/" + str(m) + ".jpg"), p)
        m += 1
    with open("data/check/sample.txt","w",encoding='utf-8') as ff:
        ff.write(str(label_list_sample) + "\n")

    logger.debug("成功加载[%d]张小图作为一个批次到内存中", len(image_list_sample))
    #logger.debug("加载到内存中一个批次的小图的标签:%s", label_list_sample)

    # 旋转做样本平衡
    image_list_rotate, label_list_rotate = rotate_to_0(image_list_sample, label_list_sample)
    image_list_all, label_list_all = rotate_and_balance(image_list_rotate, label_list_rotate)
    logger.debug("旋转并做样本均衡后，加载[%s]张小图作为一个批次到内存中", len(label_list_all))
    return image_list_all, label_list_all


def rotate_to_0(image_list_sample,label_list_sample):
    '''
    将所有抽取的小图旋转正
    '''
    label_list_rotate = []
    image_list_rotate = []
    arr = np.array(label_list_sample)

    if 0 in label_list_sample:
        index0 = np.where(arr == 0)
        for l in index0[0]:
            img = image_list_sample[l]
            l = 0
            label_list_rotate.append(l)
            image_list_rotate.append(img)

    if 1 in label_list_sample:
        index1 = np.where(arr == 1)
        for i in index1[0]:
            img = image_list_sample[i]
            img = rotate(img, -90, scale=1.0)
            i = 0
            label_list_rotate.append(i)
            image_list_rotate.append(img)

    if 2 in label_list_sample:
        index2 = np.where(arr == 2)
        for j in index2[0]:
            img = image_list_sample[j]
            img = rotate(img, 180, scale=1.0)
            j = 0
            label_list_rotate.append(j)
            image_list_rotate.append(img)

    if 3 in label_list_sample:
        index3 = np.where(arr == 3)
        for k in index3[0]:
            img = image_list_sample[k]
            img = rotate(img, 90, scale=1.0)
            k = 0
            label_list_rotate.append(k)
            image_list_rotate.append(img)

    #image_list_rotate = np.stack(image_list_rotate, axis=0)
    logger.debug("统一旋转正后加载[%s]张小图作为一个批次到内存中", len(label_list_rotate))
    return image_list_rotate, label_list_rotate

def rotate_and_balance(image_list_rotate, label_list_rotate):
    '''
    统一旋转做样本均衡
    '''
    image_list_all = []
    label_list_all = []
    for img in image_list_rotate:
        img_rotate_1 = rotate(img, 90, scale=1.0)
        img_rotate_2 = rotate(img, 180, scale=1.0)
        img_rotate_3 = rotate(img, 270, scale=1.0)
        image_list_all.append(img)
        image_list_all.append(img_rotate_1)
        image_list_all.append(img_rotate_2)
        image_list_all.append(img_rotate_3)
        label_0 = 0
        label_1 = 1
        label_2 = 2
        label_3 = 3
        label_list_all.append(label_0)
        label_list_all.append(label_1)
        label_list_all.append(label_2)
        label_list_all.append(label_3)

        # show(img, str(label_0))
        # show(img_rotate_1, str(label_1))
        # show(img_rotate_2, str(label_2))
        # show(img_rotate_3, str(label_3))

    #image_list_all = np.stack(image_list_all, axis=0)
    i = 0
    for p in image_list_all:
        cv2.imwrite(os.path.join("data/check/train/" + str(i) + ".jpg"),p)
        i +=1

    #logger.debug("旋转并做样本均衡后，加载小图作为一个批次到内存中:%s", label_list_all)
    #logger.debug("旋转并做样本均衡后，加载小图作为一个批次到内存中:%s", len(label_list_all))
    return image_list_all, label_list_all


def generator(label_file, batch_num):
    image_label_list = load_data(label_file)
    while True:
        np.random.shuffle(image_label_list)
        logger.debug("shuffle了所有的图片和标签")
        for i in range(0, len(image_label_list), batch_num):
            batch = image_label_list[i:i + batch_num]
            logger.debug("获得批次数量(%d)：从%d到%d的图片/标签的名字，准备加载...", batch_num, i, i + batch_num)
            yield _load_batch_image_labels(batch)


def get_batch(num_workers, label_file, batch_num, **kwargs):
    try:
        # 这里又藏着一个generator，注意，这个函数get_batch()本身就是一个generator
        # 但是，这里，他的肚子里，还藏着一个generator()
        # 这个generator实际上就是真正去读一张图片，返回回来了
        enqueuer = GeneratorEnqueuer(generator(label_file, batch_num, **kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=32, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                # logger.debug("开始读取缓冲队")
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    logger.debug("从GeneratorEnqueuer的queue中取出的图片")
                    break
                else:
                    # logger.debug("queue is empty, which cause we are waiting....")
                    time.sleep(0.01)
            # yield一调用，就挂起，等着外面再来调用next()了
            # 所以，可以看出来queue.get()出来的是一个图片，验证了我的想法，就是一张图，不是多张
            yield generator_output

            generator_output = None
    except BaseException as e:
        traceback.format_exc()
        logger.error("读取图片出现异常：", str(e))
    finally:
        logger.info("训练进程退出读样本循环")
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    init_logger()
    # gen = get_batch(num_workers=1,batch_num=10,label_file="data/train.txt")
    # while True:
    gen = generator(label_file="data/train.txt",batch_num=3)
    image, bbox = next(gen)
    print('done')
