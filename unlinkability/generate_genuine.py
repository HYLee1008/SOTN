import tensorflow as tf
import yaml
import numpy as np
import os
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from src.funcs import linear
from src.model import get_embd, get_args
from src.youtubeface import load_ytf_data
from src.wrapper_basicImg import wrapper_basicImg


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    m = 512
    q = 32
    beta = 9.
    batch_size = 256
    class_num = 1595
    args = get_args()

    ### Fold
    fold = tf.placeholder(tf.string)

    ### Get image and label from tfrecord
    image, label, iterator = {}, {}, {}
    image['train'], label['train'], iterator['train'] = load_ytf_data(batch_size, 'train')
    image['gallery'], label['gallery'], iterator['gallery'] = load_ytf_data(batch_size, 'train', eval=True)
    image['test'], label['test'], iterator['test'] = load_ytf_data(batch_size, 'test')

    ### Get eer evaluation dataset. Wrapper
    wrapper = wrapper_basicImg('YTF')

    ### Backbone network (Arcface)
    embedding_tensor = tf.placeholder(name='img_inputs', shape=[None, 512], dtype=tf.float32)
    labels = tf.placeholder(name='label', shape=[None, ], dtype=tf.int32)

    ### My implementation (DIom algorithm)
    with tf.variable_scope('DIom'):
        fc1 = linear(tf.nn.relu(embedding_tensor), 1024, 'fc1')
        fc2 = linear(tf.nn.relu(fc1), 1024, 'fc2')
        fc3 = linear(tf.nn.relu(fc2), m * q, 'fc3')

        h_k = tf.reshape(fc3, [-1, m, q])
        h_k = tf.nn.softmax(beta * h_k, axis=2)

        index_matrix = tf.range(1, q + 1, dtype=tf.float32)
        h = tf.reduce_sum(h_k * index_matrix, axis=2)
        h = tf.reshape(h, [-1, m])
        h_norm = tf.math.l2_normalize(h, axis=1)

    ### Graph & load setup
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True

    ### Graph & load setup
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    t_vars = tf.global_variables()
    restore_vars = [var for var in t_vars if 'embd_extractor' in var.name]

    with tf.Session(config=tf_config) as sess:
        t_vars = tf.global_variables()
        restore_DIom_vars = [var for var in t_vars if 'DIom' in var.name]

        saver = tf.train.Saver(var_list=restore_DIom_vars)
        saver.restore(sess, './model/DIom_layer')


        dist_list = []
        while wrapper.samples_left > 0:
            imgs, lbls = wrapper.get_next_batch(100)

            imgs = np.reshape(imgs, [-1, 512])

            eer_dict = {
                embedding_tensor: imgs
            }

            code = sess.run(h_norm, feed_dict=eer_dict)
            code = np.reshape(code, [-1, 2, m])

            distance = np.sum(np.prod(code, axis=1), axis=1)

            if dist_list == []:
                dist_list = distance
                label_list = lbls
            else:
                dist_list = np.concatenate((dist_list, distance), axis=0)
                label_list = np.concatenate((label_list, lbls), axis=0)

        wrapper.samples_left = np.size(wrapper.labels, axis=0)
        wrapper.next_batch_pointer = 0


        mated_list = []

        for i, lbl in enumerate(label_list):
            if lbl == 1:
                mated_list.append(dist_list[i])

        np.savetxt('./unlinkability/genuine.txt', mated_list)
