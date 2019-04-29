import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np
import sys
sys.path.append("..")
from nets import vgg

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    ph_input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='ph_input_image')
    # 接2个全连接网络
    with slim.arg_scope(vgg.vgg_arg_scope()):
        vgg_fc2 = vgg.vgg16(ph_input_image)

    restore_op = slim.assign_from_checkpoint_fn("../data/vgg_16.ckpt",
                                   slim.get_trainable_variables(),
                                   ignore_missing_vars=True)


    init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
    init_biases = tf.constant_initializer(0.0)
    w_fc1 = tf.get_variable("w_fc1", [4096,256], initializer=init_weights)
    w_b1 = tf.get_variable("w_b1", [256], initializer=init_biases)
    w_fc2 = tf.get_variable("w_fc2", [256, 4], initializer=init_weights)
    w_b2 = tf.get_variable("w_b2", [4],initializer=init_biases)

    # 接2个全连接网络
    vgg_fc2 = tf.squeeze(vgg_fc2,[1,2]) #把[1,1,4096,4096] => [4096,4096]，[1,2]，而不是[0,1,2]是因为0是batch
    vgg_fc2 = tf.Print(vgg_fc2,[tf.shape(vgg_fc2)],"vgg_fc2的shape是：")
    fc1 = tf.add(tf.matmul(vgg_fc2,w_fc1),w_b1)
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, keep_prob=0.75)
    fc2 = tf.add(tf.matmul(fc1,w_fc2),w_b2)
    fc2 = tf.nn.relu(fc2)
    classes = tf.argmax(tf.nn.softmax(fc2))

    sess.run(tf.global_variables_initializer())
    restore_op(sess)
    sess.run([fc2,classes],feed_dict={ph_input_image:np.zeros((1,224,224,3))})


# from tensorflow.contrib.slim.nets import vgg
# 这个是完全用slim中的人家定义好的vgg模型，
# 我用224x224的图片可以，比例差一点也没事，但是差的太多，比如【224, 2250】就报错说：
# InvalidArgumentError (see above for traceback): Can not squeeze dim[2], expected a dimension of 1, got 64
# 参考的这个：https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/
# 这个代码留着，以后备用
# image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='ph_input_image')
# logits, _ = vgg.vgg_16(image, num_classes=4, is_training=True, dropout_keep_prob=0.75)
# variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
# restore_op = tf.contrib.framework.assign_from_checkpoint_fn("../data/vgg_16.ckpt", variables_to_restore)
# fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
# fc8_init = tf.variables_initializer(fc8_variables)
# with tf.Session() as sess:
#     restore_op(sess)
#     sess.run(fc8_init)
#     sess.run([logits], feed_dict={image: np.zeros((1, 224, 2250, 3))})