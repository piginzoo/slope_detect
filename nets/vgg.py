import logging

import tensorflow as tf

logger = logging.getLogger("vgg")

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


# 把图片扔给VGG16，
def vgg16(inputs, scope='vgg_16'):
    logger.debug("输入数据shape=(%r)", inputs.get_shape())
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            print("inputs========= ", inputs)
            # 这里没说，其实核是[3,3,3]，多出来的3是颜色3通道
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            # 出来的是m x n x 64维度的图像
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # 出来的是m/2 x n/2 x 64维度的图像，
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            # 出来的是m/2 x n/2 x 128维度的图像，核其实是[64,3,3]，64是上层的feature map的深度
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # 出来的是m/4 x n/4 x 128维度的图像
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            # 出来的是m/4 x n/4 x 256维度的图像，核其实是[128,3,3]
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # 出来的是m/8 x n/8 x 256维度的图像
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            # 出来的是m/8 x n/8 x 512维度的图像，核其实是[256,3,3]
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # 出来的是m/16 x n/16 x 512维度的图像
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            # 出来的是m/16 x n/16 x 512维度的图像
            # 细节，最后这个单元，并没有再继续到1024个核，而是还是继续用512个核
            # 最终，出来的图像是 （m/16 x n/16 x 512）

            # 这个是标准的vgg16，没问题的，保留
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # 未指定宽高？怎么得到？
            # net = sppnet.SppNet(net, sppnet.spatial_pool_size)

            net = slim.conv2d(net, 4096, [32, 32], padding='VALID', scope='fc6')
            net = slim.dropout(net, 0.5, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = tf.Print(net, [tf.shape(net)])

            # net = slim.dropout(net, 0.5,scope='dropout7')
            # net = slim.conv2d(net, 1000, [1, 1],activation_fn=None,normalizer_fn=None,scope='fc8')

            logger.debug("VGG网络输出Shape(%r)", net.get_shape())

    return net


# 这个是有问题的代码，fc6/7用的是full_connected，导致了最终的维度不对，这里仅作保留
# 这些名字vgg_16,conv1,pool4...都要留着，因为vgg16的人家训练好的模型，就用的这些字符串当做变量的查找用
def vgg16_2(inputs, scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, 0.5, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            return net


# 注意这里，最后的FC不是靠slim.fully_connected，而是靠1x1的卷积核实现的，诡异哈，好像论文里vgg也是这样，我觉得这个代码结构比上面的靠谱
# 这个代码还可以，没啥大问题，是比较简洁的实现，留作参考
def vgg16_3(inputs, scope='vgg_16'):
    logger.debug("输入数据shape=(%r)", inputs.get_shape())
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            net = slim.dropout(net, 0.5, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = tf.Print(net, [tf.shape(net)], "vgg fc7输出的shape")
    return net
