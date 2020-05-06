import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nets.model as model
import os
import logging
from utils import data_provider as data_provider, data_util, preprocess_utils
import cv2
import random

logger = logging.getLogger(__name__)

tf.app.flags.DEFINE_string('validate_dir', 'data/validate', '')
tf.app.flags.DEFINE_string('validate_label', 'data/pred.txt', '')
tf.app.flags.DEFINE_integer('validate_times', 10, '')

FLAGS = tf.app.flags.FLAGS


def validate(sess, cls_pred, ph_input_image):
    #### 加载验证数据,随机加载FLAGS.validate_batch张
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    image_label_all = []
    pred_classes_all = []
    validate_file = FLAGS.validate_label
    batch_num = FLAGS.validate_times

    logger.info("加载验证validate数据：%s，加载%d张", validate_file, batch_num)
    image_label_list = data_provider.load_data(validate_file)
    #print("image_label_list:", image_label_list)
    if len(image_label_list) < batch_num:
        val_image_names = image_label_list
    else:
        val_image_names = random.sample(image_label_list, batch_num)

    idx = 0
    all_cnt = len(val_image_names)
    true_cnt = 0
    loop_cnt = 0
    for img_path in val_image_names:
        #print("------------------------------")
        image_file = img_path[0]
        gt_image_label = img_path[1]

        if not os.path.exists(image_file):
            logger.warning("样本图片%s不存在", image_file)
            continue
        img = cv2.imread(image_file)
        logger.debug("加载样本图片:%s,标签为:%s", image_file, gt_image_label)
        # # TODO:将一张大图切成很多小图，再随机抽取小图灌到模型中进行训练
        image_list = preprocess_utils.get_patches(img)
        logger.info("加载图片：%r,小图：%r", img_path, len(image_list))
        input_img_list = data_util.prepare4vgg(image_list)
        classes = sess.run(cls_pred, feed_dict={
            ph_input_image: input_img_list
        })  # data[3]是图像的路径，传入sess是为了调试画图用
        logger.debug("预测结果为：%r", classes)
        show(input_img_list[0])
        counts = np.bincount(classes)
        pred_class = np.argmax(counts)
        image_label_all.append(gt_image_label)
        pred_classes_all.append(pred_class)

        loop_cnt += 1
        logger.info("总%r第[%r]次验证，预测结果：%r,参考结果：%r", all_cnt, idx, pred_class, gt_image_label)
        if str(gt_image_label) == str(pred_class):
            true_cnt += 1
        logger.info("总%r第[%r]次验证，正确条数：%r,正确率：%r", all_cnt, idx, true_cnt, true_cnt / loop_cnt)
        idx += 1
    logger.debug("一个批次验证集的预测结果为：%r", pred_classes_all)
    logger.debug("一个批次验证集的Label为：%r", image_label_all)

    # pred和label格式如:[2,1,0,1,1,3]，0-3是对应的方向，0朝上，1朝右倒，2倒立，3朝左倒
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy + accuracy_score(image_label_all, pred_classes_all)
    # precision tp / (tp + fp)
    precision = precision + precision_score(image_label_all, pred_classes_all, labels=[0, 1, 2, 3], average='micro')
    # recall: tp / (tp + fn)
    recall = recall + recall_score(image_label_all, pred_classes_all, labels=[0, 1, 2, 3], average='micro')
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1 + f1_score(image_label_all, pred_classes_all, labels=[0, 1, 2, 3], average='micro')
    # accuracy = accuracy/FLAGS.validate_times
    # precision = precision/FLAGS.validate_times
    # recall = recall/FLAGS.validate_times
    # f1 = f1/FLAGS.validate_times

    return accuracy, precision, recall, f1


def restore_model(model_dir, model_file=None):
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    _, class_pred = model.model(input_image)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(tf.global_variables())
    if model_file:
        model_file_path = os.path.join(model_dir, model_file)
        logger.debug("恢复给定名字模型：%s", model_file_path)
        saver.restore(sess, model_file_path)
    else:
        ckpt = tf.train.latest_checkpoint(model_dir)
        logger.debug("最新模型目录中最新模型文件:%s", ckpt)  # 有点担心learning rate也被恢复
        saver.restore(sess, ckpt)
    return sess, input_image, class_pred


def init_logger():
    logging.basicConfig(
        format='%(asctime)s  - %(name)s- %(lineno)d : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])


def test2():
    validate_label = "data/pred.txt"
    # image_list_val, image_label_val = load_validate_data(validate_label, 10)
    # x = 0
    # print(image_label_val)
    # #show(image_list_val[0][0])
    # #show(image_list_val[1][0])
    # #show(image_list_val[2][0])
    # # show(image_list_val[3][0])
    # # show(image_list_val[4][0])
    #
    # for idx,image_list in enumerate(image_list_val):
    #
    #     #show(image_list[0])
    #
    #     print(idx,len(image_list))
    #     img_idx=0
    #     for img in image_list:
    #         cv2.imwrite("data/output2/"+str(x) + "_"+ str(idx) +"_" + str(img_idx)+".jpg",img)
    #         img_idx+=1
    #     x+=1


def test_single():
    init_logger()
    tf.app.flags.DEFINE_boolean('debug', False, '')

    img = cv2.imread("data/test/4.jpg")
    # tf.reset_default_graph()  # 重置图表
    sess, ph_input_image, classes_pred = restore_model("model/")
    classes = sess.run(classes_pred, feed_dict={
        ph_input_image: np.array([img])
    })  # data[3]是图像的路径，传入sess是为了调试画图用
    logger.debug("预测结果为：%r", classes)

    counts = np.bincount(classes)
    pred_class = np.argmax(counts)
    print(pred_class)


def show(img, title='无标题'):
    """
    本地测试时展示图片
    :param img:
    :param name:
    :return:
    """
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/Users/yanmeima/workspace/ocr/crnn/data/data_generator/fonts/simhei.ttf')
    plt.title(title, fontsize='large', fontweight='bold', FontProperties=font)
    plt.imshow(img)
    plt.show()


def main():
    init_logger()
    tf.app.flags.DEFINE_boolean('debug', False, '')
    tf.app.flags.DEFINE_string('gpu', '0', '')  # 使用第#1个GPU

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    # "rotate-2020-04-29-22-34-44-2001.ckpt"
    # tf.reset_default_graph()  # 重置图表
    sess, input_image, classes_pred = restore_model("model/")
    validate(sess, classes_pred, input_image)


if __name__ == '__main__':
    # main()
    test_single()