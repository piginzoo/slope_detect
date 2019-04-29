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
    tf.app.flags.DEFINE_boolean('debug', True, '')
    tf.app.flags.DEFINE_boolean('evaluate', True, '') # 是否进行评价（你可以光预测，也可以一边预测一边评价）
    tf.app.flags.DEFINE_boolean('split', True, '')    # 是否对小框做出评价，和画到图像上
    tf.app.flags.DEFINE_string('test_dir', '', '') # 被预测的图片目录
    tf.app.flags.DEFINE_string('image_name','', '') # 被预测的图片名字，为空就预测目录下所有的文件
    tf.app.flags.DEFINE_string('pred_dir', 'data/pred', '') # 预测后的结果的输出目录
    tf.app.flags.DEFINE_boolean('draw', True, '') # 是否把gt和预测画到图片上保存下来，保存目录也是pred_dir
    tf.app.flags.DEFINE_boolean('save', True, '') # 是否保存输出结果（大框、小框信息都要保存），保存到pred_dir目录里面去
    tf.app.flags.DEFINE_string('ctpn_model_dir', 'model/', '') # model的存放目录，会自动加载最新的那个模型
    tf.app.flags.DEFINE_string('ctpn_model_file', '', '')     # 为了支持单独文件，如果为空，就预测test_dir中的所有文件


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
        image_path = os.path.join(FLAGS.test_dir,IMAGE_PATH,FLAGS.image_name)
        logger.info("指定被检测图片：%s",image_path)
        return [image_path]

    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    images_dir = os.path.join(FLAGS.test_dir,IMAGE_PATH)
    for img_name in os.listdir(images_dir):
        for ext in exts:
            if img_name.endswith(ext):
                files.append(os.path.join(images_dir, img_name))
                break
    logger.debug('批量预测，找到需要检测的图片%d张',len(files))
    return files


# 保存预测的输出结果，保存大框和小框，都用这个函数，保存大框的时候不需要scores这个参数
def save(path, file_name,data,scores=None):
    # 输出
    with open(os.path.join(path, file_name),"w") as f:
        for i, one in enumerate(data):
            line = ",".join([str(value) for value in one])
            if scores is not None:
                line += "," + str(scores[i])
            line += "\r\n"
            f.writelines(line)
    logger.info("预测结果保存完毕：%s/%s", path, file_name)


# 定义图，并且还原模型，创建session
def initialize():
    g = tf.Graph()
    with g.as_default():
        global input_image,input_im_info,bbox_pred, cls_pred, cls_prob

        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')
        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        sess = tf.Session(graph=g,config=tf.ConfigProto(allow_soft_placement=True))
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.ctpn_model_dir)
        logger.debug("从路径[%s]查找到最新的checkpoint文件[%s]", FLAGS.ctpn_model_dir, ckpt_state)
        model_path = os.path.join(FLAGS.ctpn_model_dir, os.path.basename(ckpt_state.model_checkpoint_path))
        logger.info('从%s加载模型', format(model_path))
        saver.restore(sess, model_path)

        if FLAGS.ctpn_model_file:
            ctpn_model_file_path = os.path.join(FLAGS.ctpn_model_dir,FLAGS.ctpn_model_file)
            logger.debug("恢复给定名字的CTPN模型：%s", ctpn_model_file_path)
            saver.restore(sess,ctpn_model_file_path)
        else:
            ckpt = tf.train.latest_checkpoint(FLAGS.ctpn_model_dir)
            logger.debug("最新CTPN模型目录中最新模型文件:%s", ckpt)  # 有点担心learning rate也被恢复
            saver.restore(sess, ckpt)


    return sess


def main():
    image_name_list = get_images()
    image_list = []
    image_names = []

    for image_name in image_name_list:

        logger.info("探测图片[%s]的文字区域开始", image_name)
        try:
            img = cv2.imread(image_name)
            # 好像网络用的就是OpenCV的BGR顺序，所以也不用转了
            # img = img[:, :, ::-1]  # bgr是opencv通道默认顺序，转成标准的RGB方式
            image_list.append(img)
            image_names.append(image_name)
        except:
            print("Error reading image {}!".format(image_name))
            continue

    sess = initialize()

    pred(sess,image_list,image_names)


# image_list    : numpy数组，注意，这个格式是RGB的，如果需要使用，需要转一下[:,:,::-1]
#                 为何这么设计呢？是为了兼容Web的服务，那边传过来的是RGB顺序的。
# image_names   : 文件名字
def pred(sess,image_list,image_names):#,input_image,input_im_info,bbox_pred, cls_pred, cls_prob):

    logger.info("开始探测图片的文字区域")
    global input_image,input_im_info, bbox_pred, cls_pred, cls_prob

    # 输出的路径
    pred_draw_path = os.path.join(FLAGS.pred_dir, PRED_DRAW_PATH)
    pred_gt_path = os.path.join(FLAGS.pred_dir, PRED_GT_PATH)
    pred_bbox_path = os.path.join(FLAGS.pred_dir, PRED_BBOX_PATH)
    label_path = os.path.join(FLAGS.test_dir, LABEL_PATH)
    split_path = os.path.join(FLAGS.test_dir, SPLIT_PATH)

    if not os.path.exists(pred_bbox_path): os.makedirs(pred_bbox_path)
    if not os.path.exists(pred_draw_path): os.makedirs(pred_draw_path)
    if not os.path.exists(pred_gt_path): os.makedirs(pred_gt_path)

    result = []
    for i in range(len(image_list)):
        img = image_list[i]
        image_name = image_names[i]
        _image = {}
        _image['name'] = image_name

        logger.info("探测图片[%s]的文字区域开始",image_name)
        start = time.time()

        boxes, scores, textsegs = predict_by_network(
            sess,
            bbox_pred,
            cls_prob,
            input_im_info,
            input_image,
            img)

        _image['boxes'] = boxes

        cost_time = (time.time() - start)
        logger.info("探测图片[%s]的文字区域完成，耗时: %f" ,image_name, cost_time)


    return result



if __name__ == '__main__':

    init_params()

    if not os.path.exists(FLAGS.test_dir):
        logger.error("要识别的图片的目录[%s]不存在",FLAGS.test_dir)
        exit()
    if FLAGS.image_name and not os.path.exists(os.path.join(FLAGS.test_dir,IMAGE_PATH,FLAGS.image_name)):
        logger.error("要识别的图片[%s]不存在",os.path.join(FLAGS.test_dir,IMAGE_PATH,FLAGS.image_name))
        exit()
    if not os.path.exists(FLAGS.ctpn_model_dir):
        logger.error("模型目录[%s]不存在",FLAGS.ctpn_model_dir)
        exit()
    if FLAGS.ctpn_model_file and not os.path.exists(os.path.join(FLAGS.ctpn_model_dir,FLAGS.ctpn_model_file+".meta")):
        logger.error("模型文件[%s]不存在",os.path.join(FLAGS.ctpn_model_dir,FLAGS.ctpn_model_file + ".meta"))
        exit()

    init_logger()
    main()
