#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    模型相关工具类
    1、 恢复模型
    2、 保存模型
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import time
import logging
from utils import preprocess_utils

'''
    用赵毅训练的模型预测图片,
    注：模型在预测的时候需要加标准化，调用preprocess_utils.get_patches()函数
'''

logger = logging.getLogger("Pred")
FLAGS = tf.app.flags.FLAGS
CLASS_NAME = [0,270,180,90]


def init_params(model_path=''):
    tf.app.flags.DEFINE_string('image_name','', '')         # 被预测的图片名字，为空就预测目录下所有的文件
    tf.app.flags.DEFINE_string('pred_dir', 'data/validate', '') # 预测后的结果的输出目录
    tf.app.flags.DEFINE_string('model_path',model_path, '')   # model的存放目录，会自动加载最新的那个模型
    #tf.app.flags.DEFINE_string('model_file',model_name, '') # 为了支持单独文件，如果为空，就预测pred_dir中的所有文件
    tf.app.flags.DEFINE_boolean('debug', False, '')
    # 这个是为了兼容
    # gunicorn -w 2 -k gevent web.api_server:app -b 0.0.0.0:8080
    tf.app.flags.DEFINE_string('worker-class', 'gevent', '')
    tf.app.flags.DEFINE_integer('workers', 2, '')
    tf.app.flags.DEFINE_string('bind', '0.0.0.0:8080', '')
    tf.app.flags.DEFINE_integer('timeout', 60, '')
    tf.app.flags.DEFINE_string('gpu', '0', '')  # 使用第#1个GPU


def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])


def get_images():
    if FLAGS.image_name:
        image_path = os.path.join(FLAGS.pred_dir,FLAGS.image_name)
        logger.info("指定被检测图片：%s",image_path)
        return [image_path]

    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    images_dir = os.path.join(FLAGS.pred_dir)
    for img_name in os.listdir(images_dir):
        for ext in exts:
            if img_name.endswith(ext):
                files.append(os.path.join(images_dir, img_name))
                break
    logger.debug('批量预测，找到需要检测的图片%d张',len(files))
    return files


def main():
    with tf.Session() as sess:
        # 从pb模型直接恢复
        # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # init = tf.global_variables_initializer()
        # sess.run(init)
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.model_path)
        signature = meta_graph_def.signature_def
        in_tensor_name = signature['serving_default'].inputs['x'].name
        out_tensor_name = signature['serving_default'].outputs['predCls'].name

        input_x = sess.graph.get_tensor_by_name(in_tensor_name)
        output = sess.graph.get_tensor_by_name(out_tensor_name)

        image_name_list_all = get_images()
        lines = []
        arr_split = np.array_split(image_name_list_all,16)
        for image_name_list in arr_split:
            logger.info("批次处理：%r", len(image_name_list))

            for image_name in image_name_list:
                logger.info("探测图片[%s]开始", image_name)
                try:
                    img = cv2.imread(image_name)
                    # # TODO:将一张大图切成很多小图，直接把小图灌到模型中进行预测
                    image_list = preprocess_utils.get_patches(img)
                    logger.debug("将图像分成%d个patches", len(image_list))
                except:
                    print("Error reading image {}!".format(image_name))
                    continue

                classes = pred_pb(sess, output, input_x, image_list)
                # TODO:预测出来多个小图的标签，取众数作为大图的标签
                #print("classes:",classes)
                counts = np.bincount(classes)
                pred_classes = np.argmax(counts)

                logger.info("图片[%s]旋转角度为[%s]度", image_name, CLASS_NAME[pred_classes])
                line = image_name + " " + str(CLASS_NAME[pred_classes])
                lines.append(line)

    with open("data/pred_zhao_0528.txt", "w", encoding='utf-8') as f:
        for line in lines:
            f.write(str(line) + '\n')


def pred_pb(sess, output, input_x, image_list):
    logger.info("开始探测图片")
    start = time.time()
    #
    # from utils import data_util
    # image_list = data_util.prepare4vgg(image_list)

    classes = sess.run(output, feed_dict={input_x: image_list})
    logger.info("探测图片完成，耗时: %f", (time.time() - start))
    return classes




if __name__ == '__main__':
    init_logger()
    init_params(model_path="model/multi_pb/20200519")

    if not os.path.exists(FLAGS.pred_dir):
        logger.error("要识别的图片的目录[%s]不存在",FLAGS.pred_dir)
        exit()
    if FLAGS.image_name and not os.path.exists(os.path.join(FLAGS.pred_dir,FLAGS.image_name)):
        logger.error("要识别的图片[%s]不存在",os.path.join(FLAGS.pred_dir,FLAGS.image_name))
        exit()
    if not os.path.exists(FLAGS.model_path):
        logger.error("模型目录[%s]不存在",FLAGS.model_path)
        exit()
    if not os.path.exists(os.path.join(FLAGS.model_path,'saved_model.pb')):
        logger.error("模型文件[%s]不存在",os.path.join(FLAGS.model_path,'saved_model.pb'))
        exit()
    # 选择GPU
    if FLAGS.gpu != "1" and FLAGS.gpu != "0":
        logger.error("无法确定使用哪一个GPU，退出")
        exit()
    logger.info("使用GPU%s显卡进行训练", FLAGS.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


    param_dict = {
        'inputs': {'x':'input_data'},
        'output': {'predCls':'output'}
    }

    main()