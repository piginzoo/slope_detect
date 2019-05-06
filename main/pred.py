# coding=utf-8
import os,sys
import time
import cv2
import tensorflow as tf
import logging
sys.path.append(os.getcwd())
import nets.model as model

logger = logging.getLogger("Train")
FLAGS = tf.app.flags.FLAGS

def init_params():
    tf.app.flags.DEFINE_string('image_name','', '') # 被预测的图片名字，为空就预测目录下所有的文件
    tf.app.flags.DEFINE_string('pred_dir', 'data/pred', '') # 预测后的结果的输出目录
    tf.app.flags.DEFINE_string('model_dir', 'model/', '') # model的存放目录，会自动加载最新的那个模型
    tf.app.flags.DEFINE_string('model_file', '', '')     # 为了支持单独文件，如果为空，就预测pred_dir中的所有文件


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
    g = tf.Graph()
    with g.as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        fc2, classes = model.model(input_image)
        return fc2,classes

def restore_session():
    saver = tf.train.Saver()
    sess = tf.Session(graph=g,config=tf.ConfigProto(allow_soft_placement=True))
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
    logger.debug("从路径[%s]查找到最新的checkpoint文件[%s]", FLAGS.model_dir, ckpt_state)
    model_path = os.path.join(FLAGS.model_dir, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('从%s加载模型', format(model_path))
    saver.restore(sess, model_path)
    if FLAGS.model_file:
        model_file_path = os.path.join(FLAGS.model_dir,FLAGS.model_file)
        logger.debug("恢复给定名字的模型：%s", model_file_path)
        saver.restore(sess,model_file_path)
    else:
        ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
        logger.debug("最新模型目录中最新模型文件:%s", ckpt)  # 有点担心learning rate也被恢复
        saver.restore(sess, ckpt)
    return sess


def main():
    image_name_list = get_images()
    image_list = []
    image_names = []

    for image_name in image_name_list:

        logger.info("探测图片[%s]开始", image_name)
        try:
            img = cv2.imread(image_name)
            # 好像网络用的就是OpenCV的BGR顺序，所以也不用转了
            # img = img[:, :, ::-1]  # bgr是opencv通道默认顺序，转成标准的RGB方式
            image_list.append(img)
            image_names.append(image_name)
        except:
            print("Error reading image {}!".format(image_name))
            continue

    fc2,classes = init_model()
    sess = restore_session()

    pred(sess,image_list,image_names)


def pred(sess,classes,input_images,image_list):#,input_image,input_im_info,bbox_pred, cls_pred, cls_prob):
    logger.info("开始探测图片")
    for i in range(len(image_list)):
        start = time.time()
        _classes = sess.run(classes,feed_dict={input_images: image_list})
        logger.info("探测图片[%s]完成，耗时: %f", (time.time() - start))
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
