# coding=utf-8
import os,sys
import time
import cv2
import tensorflow as tf
import logging
sys.path.append(os.getcwd())
import nets.model as model
from utils import data_util
import numpy as np
from utils import preprocess_utils

'''
    旋转模型预测,老模型和新模型
'''

logger = logging.getLogger(__name__)

FLAGS = tf.app.flags.FLAGS
NEW_CLASS_NAME = [0,90,180,270]
OLD_CLASS_NAME = [0,270,180,90]

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


# 定义图，并且还原模型，创建session
def init_model():
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    _, classes = model.model(input_image)
    return input_image,classes


def restore_session():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(tf.global_variables())
    if not os.path.exists(FLAGS.model_dir):
        logger.error("目录%s不存在，加载模型失败")
        return None
    logger.info('从目录%s加载模型', format(FLAGS.model_dir))

    if FLAGS.model_file:
        model_file_path = os.path.join(FLAGS.model_dir,FLAGS.model_file)
        logger.debug("恢复给定名字模型：%s", model_file_path)
        if not os.path.exists(model_file_path+".meta"):
            logger.error("模型文件%s不存在，加载模型失败",model_file_path+".meta")
            return None
        saver.restore(sess,model_file_path)
    else:
        ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
        logger.debug("最新模型目录中最新模型文件:%s", ckpt)  # 有点担心learning rate也被恢复
        saver.restore(sess, ckpt)
    return sess


def main_new():
    tf.reset_default_graph()  # 重置图表
    input_images, classes = init_model()
    sess = restore_session()
    saveModDir = 'model/pb/100001'

    # 保存转换训练好的模型
    builder = tf.saved_model.builder.SavedModelBuilder(saveModDir)
    inputs = {
        "x": tf.saved_model.utils.build_tensor_info(input_images)
    }
    # B方案.直接输出一个整个的SparseTensor
    output = {
        "predCls": tf.saved_model.utils.build_tensor_info(classes),
    }

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=output,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={ # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        }
    )
    builder.save()
    print("转换模型结束", saveModDir)



if __name__ == '__main__':
    init_logger()
    init_params(model_name="rotate-2020-05-15-11-54-22-16101.ckpt")
    
    if not os.path.exists(FLAGS.model_dir):
        logger.error("模型目录[%s]不存在",FLAGS.model_dir)
        exit()
    if FLAGS.model_file and not os.path.exists(os.path.join(FLAGS.model_dir,FLAGS.model_file+".meta")):
        logger.error("模型文件[%s]不存在",os.path.join(FLAGS.model_dir,FLAGS.model_file + ".meta"))
        exit()
    # 选择GPU
    if FLAGS.gpu != "1" and FLAGS.gpu != "0":
        logger.error("无法确定使用哪一个GPU，退出")
        exit()
    logger.info("使用GPU%s显卡进行训练", FLAGS.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    # 用创哥大图训练的模型预测
    # main_old()
    # 用延美切小图训练的模型预测
    main_new()
