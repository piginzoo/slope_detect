import datetime
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from nets import model as model
from utils import data_provider as data_provider
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from utils import data_util
import logging

tf.app.flags.DEFINE_string('name','rotate_detect', '')
tf.app.flags.DEFINE_float('learning_rate', 0.01, '') #学习率
tf.app.flags.DEFINE_integer('max_steps', 40000, '') #我靠，人家原来是50000的设置
tf.app.flags.DEFINE_integer('decay_steps', 2000, '')#？？？
tf.app.flags.DEFINE_integer('evaluate_steps',10, '')#？？？
tf.app.flags.DEFINE_float('decay_rate', 0.5, '')#？？？
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('train_dir','data/train','')
tf.app.flags.DEFINE_string('train_label','data/train.txt','')
tf.app.flags.DEFINE_integer('train_batch',30,'')
tf.app.flags.DEFINE_string('validate_dir','data/validate','')
tf.app.flags.DEFINE_string('validate_label','data/validate.txt','')
tf.app.flags.DEFINE_integer('validate_batch',32,'')
tf.app.flags.DEFINE_integer('validate_times',10,'')
tf.app.flags.DEFINE_integer('early_stop',5,'')
tf.app.flags.DEFINE_integer('num_readers', 2, '')#同时启动的进程2个
tf.app.flags.DEFINE_string('gpu', '1', '') #使用第#1个GPU
tf.app.flags.DEFINE_string('model', 'model', '')
tf.app.flags.DEFINE_float('lambda1', 1000, '')
tf.app.flags.DEFINE_string('logs_path', 'logs', '')
tf.app.flags.DEFINE_string('pretrained_model_path', 'data/vgg_16.ckpt', '')#VGG16的预训练好的模型，这个是直接拿来用的
tf.app.flags.DEFINE_boolean('restore', False, '')
tf.app.flags.DEFINE_boolean('debug', False, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2000, '')
tf.app.flags.DEFINE_integer('max_width',1200,'')
tf.app.flags.DEFINE_integer('max_height',1600,'')


FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

logger = logging.getLogger("Train")


def init_logger():
    level = logging.DEBUG
    if(FLAGS.debug):
        level = logging.DEBUG

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=level,
        handlers=[logging.StreamHandler()])


def main(argv=None):

    # 选择GPU
    if FLAGS.gpu!="1" and FLAGS.gpu!="0":
        logger.error("无法确定使用哪一个GPU，退出")
        exit()
    logger.info("使用GPU%s显卡进行训练",FLAGS.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger.info(
        "本次使用的参数：\nlearning_rate:%f\ndecay_steps:%f\nmax_steps:%d\nevaluate_steps:%d\nmodel:%s\nlambda1:%d\nlogs_path:%s\nrestore:%r\ndebug:%r\nsave_checkpoint_steps:%d", \
        FLAGS.learning_rate,
        FLAGS.decay_steps,
        FLAGS.max_steps,
        FLAGS.evaluate_steps,
        FLAGS.model,
        FLAGS.lambda1,
        FLAGS.logs_path,
        FLAGS.restore,
        FLAGS.debug,
        FLAGS.save_checkpoint_steps)

    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(FLAGS.logs_path, StyleTime))
    if not os.path.exists(FLAGS.model):
        os.makedirs(FLAGS.model)


    # 输入图像数据的维度[批次,  高度,  宽度,  3通道]
    ph_input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='ph_input_image')
    ph_label = tf.placeholder(tf.int64,   shape=[None], name='ph_label')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)

    tf.summary.scalar('learning_rate', learning_rate)
    adam_opt = tf.train.AdamOptimizer(learning_rate) # 默认是learning_rate是0.001，而且后期会不断的根据梯度调整，一般不用设这个数，所以我索性去掉了

    # gpu_id = int(FLAGS.gpu)
    # with tf.device('/gpu:%d' % gpu_id):
    #     with tf.name_scope('model_%d' % gpu_id) as scope:
    cls_prob,cls_preb = model.model(ph_input_image)
    cross_entropy = model.loss(cls_prob,ph_label)
    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    #计算梯度
    grads = adam_opt.compute_gradients(cross_entropy)
    # logger.info("计算图定义完毕，定义在gpu:%d上", gpu_id)
    # 使用计算得到的梯度来更新对应的variable
    apply_gradient_op = adam_opt.apply_gradients(grads, global_step=global_step)

    # 这个是定义召回率、精确度和F1
    v_recall = tf.Variable(0.001, trainable=False)
    v_precision = tf.Variable(0.001, trainable=False)
    v_accuracy = tf.Variable(0.001, trainable=False)
    v_f1 = tf.Variable(0.001, trainable=False)
    tf.summary.scalar("Recall",v_recall)
    tf.summary.scalar("Precision",v_precision)
    tf.summary.scalar("Accuracy", v_accuracy)
    tf.summary.scalar("F1",v_f1)
    summary_op = tf.summary.merge_all()
    logger.info("summary定义完毕")

    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 某些操作执行的依赖关系，这时我们可以使用tf.control_dependencies()来实现
    # 我依赖于
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op') # no_op啥也不干，但是它依赖的操作都会被干一遍

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.logs_path,StyleTime), tf.get_default_graph())

    if FLAGS.pretrained_model_path is not None:
        logger.info('加载vgg模型：%s',FLAGS.pretrained_model_path)
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    # 早停用的变量
    #best_f1 = 0
    best_accuracy = 0
    early_stop_counter = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.model)
            logger.debug("最新的模型文件:%s",ckpt) #有点担心learning rate也被恢复
            saver.restore(sess, ckpt)
        else:
            logger.info("从头开始训练模型")
            sess.run(tf.global_variables_initializer())
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        logger.debug("开始加载训练数据")
        # 是的，get_batch返回的是一个generator
        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers,label_file=FLAGS.train_label,batch_num=FLAGS.train_batch)
        start = time.time()
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(start))

        logger.debug("开始训练")
        for step in range(FLAGS.max_steps):

            image_list,label_list = next(data_generator) # next(<迭代器>）来返回下一个结果
            logger.debug("成功加载图片%d张，标签%d个：",len(image_list),len(label_list))

            image_list = data_util.prepare4vgg(image_list)
            logger.debug("开始第%d步训练，运行sess.run,数据shape：%r",step,image_list.shape)

            _, summary_str,classes = sess.run([train_op, summary_op, cls_prob],
                feed_dict = {ph_input_image: image_list , ph_label: label_list}) # data[3]是图像的路径，传入sess是为了调试画图用 np.array(image_list)
            logger.info("结束第%d步训练，结束sess.run",step)
            summary_writer.add_summary(summary_str, global_step=step)

            if step!=0 and step % FLAGS.evaluate_steps == 0:
                logger.info("在第%d步，开始进行模型评估",step)

                # data[4]是大框的坐标，是个数组，8个值
                accuracy_value,precision_value,recall_value,f1_value = validate(sess,cls_preb,ph_input_image,ph_label)

                if accuracy_value>best_accuracy:
                    logger.info("新accuracy值[%f]大于过去最好的accuracy值[%f]，早停计数器重置",accuracy_value,best_accuracy)
                    best_accuracy = accuracy_value
                    early_stop_counter = 0
                    # 每次效果好的话，就保存一个模型
                    filename = ('ctpn-{:s}-{:d}'.format(train_start_time,step + 1) + '.ckpt')
                    filename = os.path.join(FLAGS.model, filename)
                    saver.save(sess, filename)
                    logger.info("在第%d步，保存了最好的模型文件：%s，Faccuracy：%f",step,filename,best_accuracy)
                else:
                    logger.info("新accuracy值[%f]小于过去最好的accuracy值[%f]，早停计数器+1", accuracy_value, best_accuracy)
                    early_stop_counter+= 1

                # 更新accuracy,Recall和Precision
                sess.run([tf.assign(v_f1,       f1_value),
                          tf.assign(v_recall,   recall_value),
                          tf.assign(v_precision,precision_value),
                          tf.assign(v_accuracy, accuracy_value)])
                logger.info("在第%d步，模型评估结束", step)

                if early_stop_counter> FLAGS.early_stop:
                    logger.warning("达到了早停计数次数：%d次，训练提前结束",early_stop_counter)
                    break

            if step != 0 and step % FLAGS.decay_steps == 0:
                logger.info("学习率(learning rate)衰减：%f=>%f",learning_rate.eval(),learning_rate.eval() * FLAGS.decay_rate)
                sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))


# 这里，只验证一个batch就完事，太多影响训练
# def validate(sess,cls_pred,ph_input_image,ph_label):
#
#     #### 加载验证数据,随机加载FLAGS.validate_batch张
#
#     image_list, image_label = data_provider.load_validate_data(FLAGS.validate_label,FLAGS.validate_batch)
#     logger.debug("加载了验证集%d张",len(image_list))
#
#     classes = sess.run(cls_pred,feed_dict={
#         ph_input_image:  data_util.prepare4vgg(image_list),
#         ph_label:        image_label
#     })  # data[3]是图像的路径，传入sess是为了调试画图用
#
#     logger.debug("预测结果为：%r",classes)
#     logger.debug("Label为：%r",image_label)
#
#     # pred和label格式如:[2,1,0,1,1,3]，0-3是对应的方向，0朝上，1朝右倒，2倒立，3朝左倒
#     # accuracy: (tp + tn) / (p + n)
#     accuracy = accuracy_score(image_label, classes)
#     # precision tp / (tp + fp)
#     precision = precision_score(image_label, classes,labels=[0,1,2,3],average='micro')
#     # recall: tp / (tp + fn)
#     recall = recall_score(image_label, classes,labels=[0,1,2,3],average='micro')
#     # f1: 2 tp / (2 tp + fp + fn)
#     f1 = f1_score(image_label, classes,labels=[0,1,2,3],average='micro')
#
#     return accuracy,precision,recall,f1

def validate(sess,cls_pred,ph_input_image,ph_label):

    #### 加载验证数据,随机加载FLAGS.validate_batch张
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    image_label_all = []
    classes_all = []
    for step in range(FLAGS.validate_times):
        image_list, image_label = data_provider.load_validate_data(FLAGS.validate_label,FLAGS.validate_batch)
        logger.debug("加载了验证集%d张",len(image_list))

        classes = sess.run(cls_pred,feed_dict={
            ph_input_image:  data_util.prepare4vgg(image_list),
            ph_label:        image_label
        })  # data[3]是图像的路径，传入sess是为了调试画图用

        counts = np.bincount(classes)
        classes = np.argmax(counts)
        counts = np.bincount(image_label)
        image_label = np.argmax(counts)
        # logger.debug("预测结果为：%r",classes)
        # logger.debug("Label为：%r",image_label)

        image_label_all.append(image_label)
        classes_all.append(classes)
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

if __name__ == '__main__':
    init_logger()
    tf.app.run()
