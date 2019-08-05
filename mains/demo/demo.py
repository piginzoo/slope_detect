import numpy as np
import tensorflow as tf

c = tf.constant([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

d = tf.unstack(c, axis=0)

e = tf.unstack(c, axis=1)

with tf.Session() as sess:
    print(sess.run(d))

    print(sess.run(e))


# input_image_size = tf.placeholder(tf.float32, shape=[None, None], name='ph_input_image')
#
# with tf.Session() as sess:
#     # sess.run(tf.local_variables_initializer())
#     # sess.run(tf.global_variables_initializer())
#     out = sess.run(input_image_size, feed_dict={input_image_size: [[22, 33], [22, 33]]})
#     print(out)
#     print(input_image_size.get_shape())
#
#     height = input_image_size[:][0]
#     width = input_image_size[:][1]
#
#     print(height, width)


def spp_layer(input_, levels=4, name='SPP_layer', pool_type='max_pool'):
    '''
    Multiple Level SPP layer.

    Works for levels=[1, 2, 3, 6].
    '''

    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):

        for l in range(levels):

            l = l + 1
            ksize = [1, np.ceil(shape[1] / l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]

            strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]

            if pool_type == 'max_pool':
                pool = tf.nn.max_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool, (shape[0], -1), )

            else:
                pool = tf.nn.avg_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool, (shape[0], -1))
            print("Pool Level {:}: shape {:}".format(l, pool.get_shape().as_list()))
            if l == 1:

                x_flatten = tf.reshape(pool, (shape[0], -1))
            else:
                x_flatten = tf.concat((x_flatten, pool), axis=1)
            print("Pool Level {:}: shape {:}".format(l, x_flatten.get_shape().as_list()))
            # pool_outputs.append(tf.reshape(pool, [tf.shape(pool)[1], -1]))
    print(x_flatten)
    return x_flatten


ph_input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='ph_input_image')
x = tf.ones((4, 16, 16, 3))
x_sppl = spp_layer(x, 4)
