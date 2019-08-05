# coding=utf-8
import logging
import os
import sys
import time

import cv2
import tensorflow as tf

sys.path.append(os.getcwd())
import nets.model as model
from utils import data_util
import shutil

from utils import data_provider as data_provider
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

logger = logging.getLogger("Train")
FLAGS = tf.app.flags.FLAGS
CLASS_NAME = [0, 90, 180, 270]


def init_params(model_dir='model', model_name=''):
    tf.app.flags.DEFINE_string('image_name', '', '')  # 被预测的图片名字，为空就预测目录下所有的文件
    tf.app.flags.DEFINE_string('pred_dir', 'data/pred', '')  # 预测后的结果的输出目录
    tf.app.flags.DEFINE_string('model_dir', model_dir, '')  # model的存放目录，会自动加载最新的那个模型
    tf.app.flags.DEFINE_string('model_file', model_name, '')  # 为了支持单独文件，如果为空，就预测pred_dir中的所有文件
    tf.app.flags.DEFINE_boolean('debug', False, '')
    # 这个是为了兼容
    # gunicorn -w 2 -k gevent web.api_server:app -b 0.0.0.0:8080
    tf.app.flags.DEFINE_string('worker-class', 'gevent', '')
    tf.app.flags.DEFINE_integer('workers', 2, '')
    tf.app.flags.DEFINE_string('bind', '0.0.0.0:8080', '')
    tf.app.flags.DEFINE_integer('timeout', 60, '')
    tf.app.flags.DEFINE_string('gpu', '1', '')  # 使用第#1个GPU


def init_logger():
    level = logging.DEBUG
    if (FLAGS.debug):
        level = logging.DEBUG

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=level,
        handlers=[logging.StreamHandler()])


def get_labels(label_file):
    f = open(label_file, 'r')
    labels = {}
    # 从文件中读取样本路径和标签值
    # >data/train/21.png 1
    # >data/train/22.png 0
    # >data/train/23.png 2
    for line in f:
        # logger.debug("line=%s",line)
        filename, _, label = line[:-1].partition(' ')  # partition函数只读取第一次出现的标志，分为左右两个部分,[:-1]去掉回车
        if label is None or label.strip() == "":
            logger.warning("标签数据有问题，忽略：%s", line)
            continue
        label = int(label)
        labels[filename] = label
    return labels


def get_images():
    if FLAGS.image_name:
        image_path = os.path.join(FLAGS.pred_dir, FLAGS.image_name)
        logger.info("指定被检测图片：%s", image_path)
        return [image_path]

    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    images_dir = os.path.join(FLAGS.pred_dir)
    for img_name in os.listdir(images_dir):
        for ext in exts:
            if img_name.endswith(ext):
                files.append(os.path.join(images_dir, img_name))
                break
    logger.debug('批量预测，找到需要检测的图片%d张', len(files))
    return files


# 定义图，并且还原模型，创建session
def init_model():
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    _, classes = model.model(input_image)
    return input_image, classes


def restore_session():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(tf.global_variables())
    if not os.path.exists(FLAGS.model_dir):
        logger.error("目录%s不存在，加载模型失败")
        return None
    logger.info('从目录%s加载模型', format(FLAGS.model_dir))

    if FLAGS.model_file:
        model_file_path = os.path.join(FLAGS.model_dir, FLAGS.model_file)
        logger.debug("恢复给定名字模型：%s", model_file_path)
        if not os.path.exists(model_file_path + ".meta"):
            logger.error("模型文件%s不存在，加载模型失败", model_file_path + ".meta")
            return None
        saver.restore(sess, model_file_path)
    else:
        ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
        logger.debug("最新模型目录中最新模型文件:%s", ckpt)  # 有点担心learning rate也被恢复
        saver.restore(sess, ckpt)
    return sess


def load_batch_images(image_name_list, c, batch):
    image_list = []
    image_names = []
    for idx in range(c, batch):
        file_name = image_name_list[idx]
        logger.info("---------%s--------", file_name)
        try:
            img = cv2.imread(file_name)
            # 好像网络用的就是OpenCV的BGR顺序，所以也不用转了
            # img = img[:, :, ::-1]  # bgr是opencv通道默认顺序，转成标准的RGB方式
            image_list.append(img)
            image_names.append(file_name)
        except:
            print("Error reading image {}!".format(file_name))
            continue
    return image_list, image_names


def main():
    labels = get_labels("data/validate.txt")
    result_path = "data/test/"
    result_file = open(result_path + "result_file.txt", "w")
    result_sum_file = open(result_path + "result_sum_file.txt", "w")

    image_name_list = get_images()
    # image_list = []
    # for image_name in image_name_list:
    #     logger.info("探测图片[%s]开始", image_name)
    #     try:
    #         img = cv2.imread(image_name)
    #         # 好像网络用的就是OpenCV的BGR顺序，所以也不用转了
    #         # img = img[:, :, ::-1]  # bgr是opencv通道默认顺序，转成标准的RGB方式
    #         image_list.append(img)
    #     except:
    #         print("Error reading image {}!".format(image_name))
    #         continue

    input_images, classes = init_model()
    sess = restore_session()
    # 因图片多加载慢 ，按批次加载图片
    batch = 10
    lens = len(image_name_list)
    size = int(lens / batch)
    size = size + (1 if lens % batch > 0 else 0)
    logger.info("验证图片按分批次处理：%s, 总数：%s", size, lens)
    fail_images = []
    for idx in range(size):
        logger.info("---------开始第%s个批次---------", str(idx))
        start = idx * batch
        end = batch + (idx * batch)
        if idx == size - 1:
            end = end - (end - lens)

        image_list, image_names = load_batch_images(image_name_list, start, end)
        _classes = pred(sess, classes, input_images, image_list)
        for i in range(len(_classes)):
            label = labels[image_names[i]]
            label_flag = label == _classes[i]
            if not label_flag:
                fail_images.append(image_names[i])
                logger.info("图片[%s]旋转角度为[%s]度 - 标记：%s 结果：%s", image_names[i], CLASS_NAME[_classes[i]], label,
                            label_flag)
                result_file.write(image_names[i])
                result_file.write(" 【R-角度 ")
                result_file.write(str(CLASS_NAME[_classes[i]]))
                result_file.write("】 【R-标记 ")
                result_file.write(str(_classes[i]))
                result_file.write("】 【S-标记 ")
                result_file.write(str(label))
                result_file.write("】 ")
                result_file.write(str(label_flag))
                result_file.write("\n")
                result_file.flush()

                result_sum_file.write("总数")
                result_sum_file.write(str(lens))
                result_sum_file.write("  错误数")
                result_sum_file.write(str(len(fail_images)))
                result_sum_file.write("\n")
                result_sum_file.flush()
        logger.info("---------结束第%s个批次---------", str(idx))
    result_file.close()
    result_sum_file.close()
    logger.info("图片总数：%s，错误总数：%s，准备复制失败文件到指定目录", lens, len(fail_images))
    for file_path in fail_images:
        (_, fileName) = os.path.split(file_path)
        shutil.copyfile(file_path, result_path + "fail/" + fileName)


def pred(sess, classes, input_images, image_list):  # ,input_image,input_im_info,bbox_pred, cls_pred, cls_prob):
    logger.info("开始探测图片")
    start = time.time()
    image_list = data_util.prepare4vgg(image_list)
    _classes = sess.run(classes, feed_dict={input_images: image_list})
    logger.info("探测图片完成，耗时: %f", (time.time() - start))
    return _classes


def validate(sess, cls_pred, ph_input_image, ph_label):
    #### 加载验证数据,随机加载FLAGS.validate_batch张
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    for step in range(FLAGS.validate_times):
        image_list, image_label = data_provider.load_validate_data(FLAGS.validate_label, FLAGS.validate_batch)
        logger.debug("加载了验证集%d张", len(image_list))

        classes = sess.run(cls_pred, feed_dict={
            ph_input_image: data_util.prepare4vgg(image_list),
            ph_label: image_label
        })  # data[3]是图像的路径，传入sess是为了调试画图用

        logger.debug("预测结果为：%r", classes)
        logger.debug("Label为：%r", image_label)

        # pred和label格式如:[2,1,0,1,1,3]，0-3是对应的方向，0朝上，1朝右倒，2倒立，3朝左倒
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy + accuracy_score(image_label, classes)
        # precision tp / (tp + fp)
        precision = precision + precision_score(image_label, classes, labels=[0, 1, 2, 3], average='micro')
        # recall: tp / (tp + fn)
        recall = recall + recall_score(image_label, classes, labels=[0, 1, 2, 3], average='micro')
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1 + f1_score(image_label, classes, labels=[0, 1, 2, 3], average='micro')
    accuracy = accuracy / FLAGS.validate_times
    precision = precision / FLAGS.validate_times
    recall = recall / FLAGS.validate_times
    f1 = f1 / FLAGS.validate_times

    return accuracy, precision, recall, f1


if __name__ == '__main__':
    init_params()
    if not os.path.exists(FLAGS.pred_dir):
        logger.error("要识别的图片的目录[%s]不存在", FLAGS.pred_dir)
        exit()
    if FLAGS.image_name and not os.path.exists(os.path.join(FLAGS.pred_dir, FLAGS.image_name)):
        logger.error("要识别的图片[%s]不存在", os.path.join(FLAGS.pred_dir, FLAGS.image_name))
        exit()
    if not os.path.exists(FLAGS.model_dir):
        logger.error("模型目录[%s]不存在", FLAGS.model_dir)
        exit()
    if FLAGS.model_file and not os.path.exists(os.path.join(FLAGS.model_dir, FLAGS.model_file + ".meta")):
        logger.error("模型文件[%s]不存在", os.path.join(FLAGS.model_dir, FLAGS.model_file + ".meta"))
        exit()
    # 选择GPU
    if FLAGS.gpu != "1" and FLAGS.gpu != "0":
        logger.error("无法确定使用哪一个GPU，退出")
        exit()
    logger.info("使用GPU%s显卡进行训练", FLAGS.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    init_logger()
    main()
