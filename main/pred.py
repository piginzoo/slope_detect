# coding=utf-8
import os,sys
import time
import cv2
import tensorflow as tf
import logging
sys.path.append(os.getcwd())
import nets.model as model
from utils import data_util

logger = logging.getLogger("Train")
FLAGS = tf.app.flags.FLAGS
CLASS_NAME = ["无","90","180","270"]

def init_params(model_dir='model',model_name=''):
    tf.app.flags.DEFINE_string('image_name','', '')         # 被预测的图片名字，为空就预测目录下所有的文件
    tf.app.flags.DEFINE_string('pred_dir', 'data/pred', '') # 预测后的结果的输出目录
    tf.app.flags.DEFINE_string('model_dir',model_dir, '')   # model的存放目录，会自动加载最新的那个模型
    tf.app.flags.DEFINE_string('model_file',model_name, '') # 为了支持单独文件，如果为空，就预测pred_dir中的所有文件
    tf.app.flags.DEFINE_boolean('debug', False, '')
    # 这个是为了兼容
    # gunicorn -w 2 -k gevent web.api_server:app -b 0.0.0.0:8080
    tf.app.flags.DEFINE_string('worker-class', 'gevent', '')
    tf.app.flags.DEFINE_integer('workers', 2, '')
    tf.app.flags.DEFINE_string('bind', '0.0.0.0:8080', '')
    tf.app.flags.DEFINE_integer('timeout', 60, '')


def init_logger():
    level = logging.DEBUG
    if(FLAGS.debug):
        level = logging.DEBUG

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=level,
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
        if not os.path.exists(model_file_path):
            logger.error("模型文件%s不存在，加载模型失败")
            return None
        saver.restore(sess,model_file_path)
    else:
        ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
        logger.debug("最新模型目录中最新模型文件:%s", ckpt)  # 有点担心learning rate也被恢复
        saver.restore(sess, ckpt)
    return sess


def main():
    image_name_list = get_images()
    image_list = []
    for image_name in image_name_list:
        logger.info("探测图片[%s]开始", image_name)
        try:
            img = cv2.imread(image_name)
            # 好像网络用的就是OpenCV的BGR顺序，所以也不用转了
            # img = img[:, :, ::-1]  # bgr是opencv通道默认顺序，转成标准的RGB方式
            image_list.append(img)
        except:
            print("Error reading image {}!".format(image_name))
            continue
    input_images,classes = init_model()
    sess = restore_session()
    classes = pred(sess,classes,input_images,image_list)
    for i in range(len(classes)):
        logger.info("图片[%s]旋转角度为[%s]度",image_name_list[i],CLASS_NAME[classes[i]])


def pred(sess,classes,input_images,image_list):#,input_image,input_im_info,bbox_pred, cls_pred, cls_prob):
    logger.info("开始探测图片")
    start = time.time()
    image_list = data_util.prepare4vgg(image_list)
    _classes = sess.run(classes,feed_dict={input_images: image_list})
    logger.info("探测图片完成，耗时: %f", (time.time() - start))
    return _classes


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

