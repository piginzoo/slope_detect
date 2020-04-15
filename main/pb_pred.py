#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    模型相关工具类
    1、 恢复模型
    2、 保存模型
"""

import os
import tensorflow as tf


# def restore_model_by_dir(model_path, input_map, output_map):
#     """
#         从目录下寻找最新的模型加载
#     :param model_path:
#     :param input_map:
#     :param output_map:
#     :return:
#     """
#     f_list = os.listdir(model_path)
#     dirs = [i for i in f_list if os.path.isdir(os.path.join(model_path, i))]
#     max_dir = max(dirs)
#     return restore_model(os.path.join(model_path, max_dir), input_map, output_map)

# tf.get_variable_scope().reuse_variables()
# tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], real_path)

def restore_model(model_path, input_dict, output_dict):
    """
        直接指定模型
    :param model_path:
    :return:
    """
    print("恢复模型：", model_path,input_dict,output_dict)
    # tf.reset_default_graph()
    params = {}
    # g = tf.get_default_graph()
    g = tf.Graph()
    with g.as_default():
        # 从pb模型直接恢复 TODO ! 这里的config也可以从参数里传过来
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # init = tf.global_variables_initializer()
        # sess.run(init)
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
        signature = meta_graph_def.signature_def
        if input_dict:
            for input_k in input_dict:
                in_tensor_name = signature['serving_default'].inputs[input_k].name
                input_param = sess.graph.get_tensor_by_name(in_tensor_name)
                params[input_dict[input_k]] = input_param
        if output_dict:
            for output_k in output_dict:
                out_tensor_name = signature['serving_default'].outputs[output_k].name
                output_param = sess.graph.get_tensor_by_name(out_tensor_name)
                params[output_dict[output_k]] = output_param
        params["session"] = sess
        params["graph"] = g

    return params


def test1():
    param_dict = {
        'inputs': {'input_data': 'data/pred/validate'},
        'output': {'output': 'data/pred/seg_maps_pred'}
    }
    model_path = "model/pb/"
    params = restore_model(model_path, param_dict['inputs'], param_dict['output'])
    print("asdas")
    return params


if __name__ == '__main__':
    p = test1()








# import tensorflow as tf
# from tensorflow.python.platform import gfile
#
# pb_file_path = "model/"
#
# sess = tf.Session()
# with gfile.FastGFile(pb_file_path+'saved_model.pb', 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     sess.graph.as_default()
#     tf.import_graph_def(graph_def, name='') # 导入计算图
#
# # 需要有一个初始化的过程
# sess.run(tf.global_variables_initializer())
#
# # 需要先复原变量
# print(sess.run('b:0'))
# # 1
#
# # 输入
# input_x = sess.graph.get_tensor_by_name('x:0')
# input_y = sess.graph.get_tensor_by_name('y:0')
#
# op = sess.graph.get_tensor_by_name('op_to_store:0')
#
# ret = sess.run(op,  feed_dict={input_x: 5, input_y: 5})
# print(ret)
# # 输出 26

