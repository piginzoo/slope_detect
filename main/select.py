# coding=utf-8
import os,sys
import time
import cv2
import tensorflow as tf
import logging
sys.path.append(os.getcwd())
from main import pred
import shutil

logger = logging.getLogger("Select")
FLAGS = tf.app.flags.FLAGS


def init_params(model_dir='model',model_name=''):
    tf.app.flags.DEFINE_string('image_name','', '')         # 被预测的图片名字，为空就预测目录下所有的文件
    tf.app.flags.DEFINE_string('pred_dir', 'data/pred', '') # 预测后的结果的输出目录
    tf.app.flags.DEFINE_string('model_dir',model_dir, '')   # model的存放目录，会自动加载最新的那个模型
    tf.app.flags.DEFINE_string('model_file',model_name, '') # 为了支持单独文件，如果为空，就预测pred_dir中的所有文件
    tf.app.flags.DEFINE_string('target_dir','','')
    tf.app.flags.DEFINE_boolean('debug', False, '')

def init_logger():
    level = logging.DEBUG
    if(FLAGS.debug):
        level = logging.DEBUG

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=level,
        handlers=[logging.StreamHandler()])


def main():
    input_images,classes = pred.init_model()
    sess = pred.restore_session()

    image_name_list = pred.get_images()
    image_list = []

    i=0
    for image_name in image_name_list:
        start = time.time()
        logger.info("探测图片[%s]开始", image_name)
        try:
            img = cv2.imread(image_name)
            image_list.append(img)
            classes = pred(sess, classes, input_images,[img])
            if classes[0]!=0:
                select_image(image_name)
        except Exception as e:
            logger.error("处理图片[%s]发生错误：%s",image_name,str(e))
            continue
        i+=1
        logger.debug("处理完第%d张图片，耗时:%f",i,time.time()-start)

def select_image(image_name,cls):
    dst_dir = FLAGS.target_dir
    shutil.copyfile(image_name,dst_dir)
    logger.warning("这张图片是歪的[%s]，挑出来",str(pred.CLASS_NAME[cls]))

if __name__ == '__main__':
    init_params()
    if not os.path.exists(FLAGS.pred_dir):
        logger.error("要识别的图片的目录[%s]不存在",FLAGS.pred_dir)
        exit()
    if FLAGS.image_name and not os.path.exists(os.path.join(FLAGS.pred_dir,FLAGS.image_name)):
        logger.error("要识别的图片[%s]不存在",os.path.join(FLAGS.pred_dir,FLAGS.image_name))
        exit()
    if not os.path.exists(FLAGS.model_dir):
        logger.error("模型目录[%s]不存在",FLAGS.model_dir)
        exit()
    if FLAGS.model_file and not os.path.exists(os.path.join(FLAGS.model_dir,FLAGS.model_file+".meta")):
        logger.error("模型文件[%s]不存在",os.path.join(FLAGS.model_dir,FLAGS.model_file + ".meta"))
        exit()

    init_logger()
    main()

