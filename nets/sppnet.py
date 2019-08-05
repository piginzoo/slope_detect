import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger("vgg")

#### spatial pool size ####
spatial_pool_size = [4, 2, 1]


####################### spatial pool #####################
def SppNet(conv5, spatial_pool_size):
    logger.info("SPP NET ============>>>> conv5 shape:%s", conv5.get_shape())
    ############### get feature size ##############
    height = int(conv5.get_shape()[1])
    width = int(conv5.get_shape()[2])

    ############### get batch size ##############
    batch_num = int(conv5.get_shape()[0])

    for i in range(len(spatial_pool_size)):

        ############### stride ##############
        stride_h = int(np.ceil(height / spatial_pool_size[i]))
        stride_w = int(np.ceil(width / spatial_pool_size[i]))

        ############### kernel ##############
        window_w = int(np.ceil(width / spatial_pool_size[i]))
        window_h = int(np.ceil(height / spatial_pool_size[i]))

        ############### max pool ##############
        max_pool = tf.nn.max_pool(conv5, ksize=[1, window_h, window_w, 1], strides=[1, stride_h, stride_w, 1],
                                  padding='SAME')

        if i == 0:
            spp = tf.reshape(max_pool, [batch_num, -1])
        else:
            ############### concat each pool result ##############
            spp = tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [batch_num, -1])])
    logger.info("END SPP NET ============>>>>")
    return spp
