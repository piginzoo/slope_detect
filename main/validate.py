import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nets.model as model
import os
import logging
from utils import data_provider as data_provider, data_util

logger = logging.getLogger(__name__)


tf.app.flags.DEFINE_string('validate_dir','data/validate','')
tf.app.flags.DEFINE_string('validate_label','data/validate.txt','')
tf.app.flags.DEFINE_integer('validate_batch',2,'')
tf.app.flags.DEFINE_integer('validate_times',10,'')

FLAGS = tf.app.flags.FLAGS


def validate(sess,cls_pred,ph_input_image):
    #### 加载验证数据,随机加载FLAGS.validate_batch张
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    image_label_all = []
    classes_all = []
    image_list_val, image_label_val = data_provider.load_validate_data(FLAGS.validate_label, FLAGS.validate_times)
    for idx,image_list in enumerate(image_list_val):
    # for step in range(FLAGS.validate_times):
        #logger.debug("加载了验证集%d张",len(image_list))
        classes = sess.run(cls_pred,feed_dict={
            ph_input_image:  data_util.prepare4vgg(image_list)
            # ,
            # ph_label:        image_labels
        })  # data[3]是图像的路径，传入sess是为了调试画图用
        image_labels = image_label_val[idx]
        logger.debug("预测结果为：%r",classes)
        logger.debug("Label为：%r",image_labels)

        counts = np.bincount(classes)
        pred_class = np.argmax(counts)

        gt_label_counts = np.bincount(image_labels)
        gt_image_label = np.argmax(gt_label_counts)

        # # check
        # m = 0
        # for p in image_list:
        #     cv2.imwrite(os.path.join("data/check0427/check/validate/" + str(m) + ".jpg"), p)
        #     m += 1

        image_label_all.append(gt_image_label)
        classes_all.append(pred_class)
    logger.debug("一个批次验证集的预测结果为：%r", classes_all)
    logger.debug("一个批次验证集的Label为：%r", image_label_all)

    # pred和label格式如:[2,1,0,1,1,3]，0-3是对应的方向，0朝上，1朝右倒，2倒立，3朝左倒
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy + accuracy_score(image_label_all, classes_all)
    # precision tp / (tp + fp)
    precision = precision + precision_score(image_label_all, classes_all,labels=[0,1,2,3],average='micro')
    # recall: tp / (tp + fn)
    recall = recall + recall_score(image_label_all, classes_all,labels=[0,1,2,3],average='micro')
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1 + f1_score(image_label_all, classes_all,labels=[0,1,2,3],average='micro')
    # accuracy = accuracy/FLAGS.validate_times
    # precision = precision/FLAGS.validate_times
    # recall = recall/FLAGS.validate_times
    # f1 = f1/FLAGS.validate_times

    return accuracy,precision,recall,f1


def restore_model(model_dir,model_file=None):
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    _, class_pred = model.model(input_image)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(tf.global_variables())
    if model_file:
        model_file_path = os.path.join(model_dir,model_file)
        logger.debug("恢复给定名字模型：%s", model_file_path)
        saver.restore(sess,model_file_path)
    else:
        ckpt = tf.train.latest_checkpoint(model_dir)
        logger.debug("最新模型目录中最新模型文件:%s", ckpt)  # 有点担心learning rate也被恢复
        saver.restore(sess, ckpt)
    return sess, input_image, class_pred



def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])

if __name__ == '__main__':
    init_logger()
    tf.app.flags.DEFINE_boolean('debug', False, '')
    # tf.reset_default_graph()  # 重置图表
    sess, input_image, classes_pred = restore_model("model/")
    validate(sess,classes_pred,input_image)