# encoding:utf-8
import os
import time
import logging
import cv2
import numpy as np
import random
from utils.data_util import GeneratorEnqueuer
import traceback
logger = logging.getLogger("data provider")



def load_data(label_file):
    f = open(label_file, 'r')
    filenames = []
    labels = []
    # 从文件中读取样本路径和标签值
    # >data/train/21.png 1
    # >data/train/22.png 0
    # >data/train/23.png 2
    for line in f:
        # logger.debug("line=%s",line)
        filename , _ , label = line[:-1].partition(' ') # partition函数只读取第一次出现的标志，分为左右两个部分,[:-1]去掉回车
        if label is None or label.strip()=="" :
            logger.warning("标签数据有问题，忽略：%s",line)
            continue
        filenames.append(filename)
        label = int(label)
        labels.append(label)

    logger.info("最终样本标签数量[%d],样本图像数量[%d]",len(labels),len(filenames))
    return list(zip(filenames, labels))


# 按照FLAGS.validate_num 随机从目录中产生批量的数据，用于做验证集
def load_validate_data(validate_file,batch_num):
    logger.info("加载验证validate数据：%s，加载%d张", validate_file,batch_num)
    image_label_list = load_data(validate_file)
    val_image_names = random.sample(image_label_list,batch_num)
    image_list, label_list = _load_batch_image_labels(val_image_names)
    return np.array(image_list),label_list

# 加载一个批次数量的图片和标签，数量为batch数
def _load_batch_image_labels(batch):
    image_list = []
    label_list = []
    for image_label_pair in batch:  # 遍历所有的图片文件
        image_file = image_label_pair[0]
        label = image_label_pair[1]
        if not os.path.exists(image_file):
            logger.warning("样本图片%s不存在", image_file)
            continue
        image_list.append(cv2.imread(image_file))
        #logger.debug("加载了图片：%s",image_file)
        label_list.append(label)
    logger.debug("加载%d张图片作为一个批次到内存中",len(image_list))
    return image_list,label_list


def generator(label_file,batch_num):
    image_label_list = load_data(label_file)
    while True:
        np.random.shuffle(image_label_list)
        logger.debug("shuffle了所有的图片和标签")
        for i in range(0, len(image_label_list), batch_num):
            batch = image_label_list[i:i + batch_num]
            logger.debug("获得批次数量(%d)：从%d到%d的图片/标签的名字，准备加载...",batch_num,i,i + batch_num)
            yield _load_batch_image_labels(batch)


def get_batch(num_workers,label_file,batch_num,**kwargs):
    try:
        # 这里又藏着一个generator，注意，这个函数get_batch()本身就是一个generator
        # 但是，这里，他的肚子里，还藏着一个generator()
        # 这个generator实际上就是真正去读一张图片，返回回来了
        enqueuer = GeneratorEnqueuer(generator(label_file,batch_num,**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=300, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                logger.debug("开始读取缓冲队列")
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    logger.debug("从GeneratorEnqueuer的queue中取出的图片")
                    break
                else:
                    time.sleep(0.01)
            # yield一调用，就挂起，等着外面再来调用next()了
            # 所以，可以看出来queue.get()出来的是一个图片，验证了我的想法，就是一张图，不是多张
            yield generator_output

            generator_output = None
    except Exception as e:
        traceback.format_exc()
        logger.error("读取图片出现异常：",str(e))
    finally:
        logger.info("训练进程退出读样本循环")
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    gen = get_batch(num_workers=2)
    while True:
        image, bbox, im_info = next(gen)
        print('done')
