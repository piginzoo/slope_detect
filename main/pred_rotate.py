# -*- coding:utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

from utils import preprocess_utils
from tuning import RotateProcessor

import os

# import ocr.service.ocr_service

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# tf.app.flags.DEFINE_string('pred_data_path', './data/pred/input', '')
# tf.app.flags.DEFINE_string('pred_gpu_list', '1', '')
# tf.app.flags.DEFINE_string('pred_model_path', './model', '')
# tf.app.flags.DEFINE_string('output_dir', './data/pred/output', '')
# tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
#
#
# FLAGS = tf.app.flags.FLAGS

CLASS_NAME = [0, 270, 180, 90]



def restore_model(model_path):
    """
        直接指定模型
    :param model_path:
    :return:
    """
    print("恢复模型：",model_path)
    # tf.reset_default_graph()
    params={}
    g = tf.get_default_graph()
    with g.as_default():
        #从pb模型直接恢复
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # init = tf.global_variables_initializer()
        # sess.run(init)
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
        signature = meta_graph_def.signature_def
        in_tensor_name = signature['serving_default'].inputs['input_image'].name
        out_tensor_name = signature['serving_default'].outputs['output'].name

        input_x = sess.graph.get_tensor_by_name(in_tensor_name)
        output = sess.graph.get_tensor_by_name(out_tensor_name)

        params["input"] = input_x
        params["output"] = output
        params["session"] = sess
        params["graph"] = g
    return params


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    path = "data/validate"
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    return files


def resize_image(im, max_side_len=1200):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

# 调用前向运算来计算
def predict_by_network(params,patches):
    # 还原各类张量
    t_input_x = params["input"]
    output_x = params["output"]
    session = params["session"]
    g = params["graph"]

    with g.as_default():
        # logger.debug("通过session预测：%r",img.shape)
        seg_maps = session.run(output_x, feed_dict={t_input_x: patches})

    return seg_maps


def crop_image_edge(image,percent):
    """
    切除图片的边缘
    :param image:
    :param percent: 边缘百分比
    :return:
    """
    w,h,_ = image.shape
    w_c = int(w*percent)
    h_c = int(h*percent)

    new_img = image[w_c:w-w_c , h_c:h-h_c]
    return new_img


def main(argv=None):
    model_path= "./model/pb/100000"
    # g = tf.get_default_graph()
    # with g.as_default():
    with tf.Session() as sess:
        # 从pb模型直接恢复
        # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # init = tf.global_variables_initializer()
        # sess.run(init)
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
        signature = meta_graph_def.signature_def
        in_tensor_name = signature['serving_default'].inputs['x'].name
        out_tensor_name = signature['serving_default'].outputs['predCls'].name

        input_x = sess.graph.get_tensor_by_name(in_tensor_name)
        output = sess.graph.get_tensor_by_name(out_tensor_name)
        # 遍历所有图片预测
        im_fn_list = get_images()
        for im_fn in im_fn_list:
            # logger.debug('image file:{}'.format(im_fn))
            image = cv2.imread(im_fn)
            # 预测
            processor = RotateProcessor()
            angle, img_rotate = ocr.service.ocr_service.process(image)

            # img_rotate = crop_image_edge(img_rotate,0.05)
            print("小角度：",angle)
            patches = preprocess_utils.get_patches(img_rotate)
            # logger.debug("将图像分成%d个patches", len(patches))
            print("开始预测")
            candiCls = sess.run(output, feed_dict={input_x: patches})
            # candiCls =  predict_by_network(params,patches)
            # candiCls = tfserving.preprocess_tf_serving_call(patches)
            print("预测结束:",candiCls)

            # 返回众数
            counts = np.bincount(candiCls)
            cls = np.argmax(counts)

            angle = CLASS_NAME[cls]
            print(im_fn,cls,angle)
            if cls == 0:
                rotate_image = image
            elif cls == 1:
                rotate_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif cls == 2:
                rotate_image = cv2.rotate(image, cv2.ROTATE_180)
            elif cls == 3:
                rotate_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite("data/output/" +os.path.basename(im_fn),rotate_image)


if __name__ == '__main__':
    # tf.app.run()
    main()