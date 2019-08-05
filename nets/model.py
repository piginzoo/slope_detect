import logging

import tensorflow as tf
from tensorflow.contrib import slim

from nets import vgg
from utils import _p_shape

FLAGS = tf.app.flags.FLAGS

logger = logging.getLogger('model_train')


# [123.68, 116.78, 103.94] 这个是VGG的预处理要求的，必须减去这个均值：https://blog.csdn.net/smilejiasmile/article/details/80807050
def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.get_shape().as_list()[-1]  # 通道数
    logger.info("mean_image_subtraction ===========>>>>>>> images shape:%s", images.get_shape())
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    # 干啥呢？ 按通道，多分出一个维度么？
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
        # 每个通道感觉都减去了一个数，貌似标准化一样
        # 不过这个 means 是如何决定的呢？

    return tf.concat(axis=3, values=channels)  # 然后把图片再合并回来


def make_var(name, shape, initializer=None):
    return tf.get_variable(name, shape, initializer=initializer)


def model(image, image_size):
    image = _p_shape(image, "最开始输入")
    image = mean_image_subtraction(image)
    with slim.arg_scope(vgg.vgg_arg_scope()):
        # 最终，出来的图像是 （m/16 x n/16 x 512）
        vgg_fc2,pool5 = vgg.vgg16(image)
        # vgg_fc2 = _p_shape(vgg_fc2, "VGG的5-3卷基层输出")
        # vgg_fc2 = tf.squeeze(vgg_fc2, [1, 2])  # 把[1,1,4096] => [4096]，[1,2]，而不是[0,1,2]是因为0是batch

    # 照着vgg，定义我自己的全连接层
    net = slim.conv2d(pool5, 1024, [32, 32], padding='VALID', scope='slope_conv1')
    net = slim.dropout(net, 0.5, scope='slope_dropout')
    net = slim.conv2d(net, 1024, [1, 1], scope='slope_conv2')
    net = tf.squeeze(net, [1, 2])

    # 先注释掉
    init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
    init_biases = tf.constant_initializer(0.0)
    # w_fc1 = tf.get_variable("w_fc1", [4096, 256], initializer=init_weights)
    # w_b1 = tf.get_variable("w_b1", [256], initializer=init_biases)
    w_fc2 = tf.get_variable("w_fc2", [1024, 4], initializer=init_weights)
    w_b2 = tf.get_variable("w_b2", [4], initializer=init_biases)

    # 接2个全连接网络
    # fc1 = tf.add(tf.matmul(vgg_fc2, w_fc1), w_b1)
    # fc1 = tf.nn.relu(fc1)
    # fc1 = tf.nn.dropout(fc1, keep_prob=0.75)
    fc2 = tf.add(tf.matmul(net, w_fc2), w_b2)
    fc2 = tf.nn.relu(fc2)
    fc2 = _p_shape(fc2, "fc2 shape:\t")

    classes = tf.argmax(tf.nn.softmax(fc2), axis=1)
    classes = _p_shape(classes, "classes shape:\t")

    return fc2, classes


def loss(fc2, labels):
    cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fc2)
    cross_entropy = tf.reduce_mean(cross_entropy_n)
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy
