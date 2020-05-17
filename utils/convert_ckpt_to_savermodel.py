#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

'''
    ckpt模型转换为savermodel
'''

# 参考：http://blog.chinaunix.net/uid-20680966-id-5830302.html
def test(checkpoint_file, export_path):
    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(graph=graph, config=config) as sess:
        # Restore from checkpoint
        loader = tf.train.import_meta_graph(checkpoint_file + '.meta')
        loader.restore(sess, checkpoint_file)

        # Export checkpoint to SavedModel
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(sess, ['cpu_server_1'])
        builder.save()



if __name__ == '__main__':
    checkpoint_file = "model/0517/rotate-2020-05-15-11-54-22-16101.ckpt"
    export_path = "model/20200517/"

    test(checkpoint_file, export_path)