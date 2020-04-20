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
from utils import data_util
from utils import preprocess_utils

logger = logging.getLogger("pred")
FLAGS = tf.app.flags.FLAGS
CLASS_NAME = [0,90,180,270]


def init_params(model_dir='model',model_name=''):
    tf.app.flags.DEFINE_string('image_name','', '')         # 被预测的图片名字，为空就预测目录下所有的文件
    tf.app.flags.DEFINE_string('pred_dir', 'data/validate', '') # 预测后的结果的输出目录
    tf.app.flags.DEFINE_string('model_dir',model_dir, '')   # model的存放目录，会自动加载最新的那个模型
    tf.app.flags.DEFINE_string('model_file',model_name, '') # 为了支持单独文件，如果为空，就预测pred_dir中的所有文件
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

def restore_model_by_dir(model_path, input_map, output_map):
    """
        从目录下寻找最新的模型加载
    :param model_path:
    :param input_map:
    :param output_map:
    :return:
    """
    f_list = os.listdir(model_path)
    dirs = [i for i in f_list if os.path.isdir(os.path.join(model_path, i))]
    max_dir = max(dirs)
    return restore_model(os.path.join(model_path, max_dir), input_map, output_map)


def restore_model(model_path, input_dict, output_dict):
    """
        直接指定模型
    :param model_path:
    :return:
    """
    print("恢复模型：", model_path,input_dict,output_dict)
    # tf.reset_default_graph()
    params = {}
    # g = tf.get_default_graph()
    g = tf.Graph()
    with g.as_default():
        # 从pb模型直接恢复 TODO ! 这里的config也可以从参数里传过来
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # init = tf.global_variables_initializer()
        # sess.run(init)
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
        signature = meta_graph_def.signature_def
        if input_dict:
            for input_k in input_dict:
                in_tensor_name = signature['serving_default'].inputs[input_k].name
                input_param = sess.graph.get_tensor_by_name(in_tensor_name)
                params[input_dict[input_k]] = input_param
        if output_dict:
            for output_k in output_dict:
                out_tensor_name = signature['serving_default'].outputs[output_k].name
                output_param = sess.graph.get_tensor_by_name(out_tensor_name)
                params[output_dict[output_k]] = output_param
        params["session"] = sess
        params["graph"] = g
    return params


def pred(params, classes, input_images, image_list):
    logger.info("开始探测图片")
    start = time.time()
    sess = params["session"]
    image_list = data_util.prepare4vgg(image_list)
    _classes = sess.run(classes, feed_dict={input_images: image_list})
    logger.info("探测图片完成，耗时: %f", (time.time() - start))
    return _classes


def main(params):
    image_name_list_all = get_images()
    lines = []

    arr_split = np.array_split(image_name_list_all,10)
    for image_name_list in arr_split:
        logger.info("批次处理：%r", len(image_name_list))

        for image_name in image_name_list:
            logger.info("探测图片[%s]开始", image_name)
            try:
                img = cv2.imread(image_name)

                # # TODO:将一张大图切成很多小图，直接把小图灌到模型中进行预测
                image_list = preprocess_utils.get_patches(img)
                logger.debug("将图像分成%d个patches", len(image_list))
                #logger.debug("需要检测的图片[%s]",image_list)
            except:
                print("Error reading image {}!".format(image_name))
                continue
        tf.reset_default_graph()  # 重置图表

        sess = params["session"]
        classes = pred(sess, classes, np.array(image_list))

        # TODO:预测出来多个小图的标签，取众数作为大图的标签
        counts = np.bincount(classes)
        classes = np.argmax(counts)

        logger.info("图片[%s]旋转角度为[%s]度", image_name, CLASS_NAME[classes])
        line = image_name + " " + str(CLASS_NAME[classes])
        lines.append(line)

    with open("data/pred.txt", "w", encoding='utf-8') as f:
        for line in lines:
            f.write(str(line) + '\n')


def test1():
    param_dict = {
        'inputs': {'input_data': 'data/validate'},
        'output': {'output': 'data/seg_maps_pred'}
    }
    model_path = "model/pb"
    params = restore_model_by_dir(model_path, param_dict['inputs'], param_dict['output'])
    print("asdas")
    return params



if __name__ == '__main__':
    p = test1()
    main(p)