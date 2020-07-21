import tensorflow as tf
import numpy as np
import os
import yaml
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from src.model import get_embd, get_args
from src.funcs import linear
from src.youtubeface import load_ytf_data
from src.wrapper_basicImg import wrapper_basicImg


def permute(input, m, q):
    indices = np.zeros((m, q, 2))
    for i in range(m):
        index = np.arange(q)
        np.random.shuffle(index)
        index = np.stack((i * np.ones(q), index), axis=-1)

        indices[i] = index

    # index = np.arange(q)
    # np.random.shuffle(index)
    # for i in range(m):
    #     idx = np.stack((i * np.ones(q), index), axis=-1)
    #
    #     indices[i] = idx

    indices_tf = tf.convert_to_tensor(indices, dtype=tf.int32)

    output = tf.transpose(input, (1, 2, 0))
    output = tf.gather_nd(output, indices_tf)
    output = tf.transpose(output, (2, 0, 1))

    return output




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    total_iteration = 300000
    m = 512
    q = 32
    beta = 9.
    batch_size = 256
    class_num = 1595
    is_permute = True
    args = get_args()

    ### Fold
    fold = tf.placeholder(tf.string)

    ### Get image and label from tfrecord
    image, label, iterator = {}, {}, {}
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

        h_k_original = tf.reshape(fc3, [-1, m, q])
        ### permute
        if is_permute:
            h_k_permuted = permute(h_k_original, m, q)

            h_k = tf.nn.softmax(beta * h_k_permuted, axis=2)
        else:
            h_k = tf.nn.softmax(beta * h_k_original, axis=2)

        index_matrix = tf.range(1, q + 1, dtype=tf.float32)
        h = tf.reduce_sum(h_k * index_matrix, axis=2)
        h = tf.reshape(h, [-1,  m])
        h_norm = tf.math.l2_normalize(h, axis=1)

    ### Graph & load setup
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        t_vars = tf.global_variables()
        restore_DIom_vars =[var for var in t_vars if 'DIom' in var.name]

        saver = tf.train.Saver(var_list=restore_DIom_vars)
        saver.restore(sess, './model/DIom_layer')

        ### Calculate EER
        dist_list = []
        label_list = []
        while wrapper.samples_left > 0:
            imgs, lbls = wrapper.get_next_batch(100)

            imgs = np.reshape(imgs, [-1, 512])

            eer_dict = {
                embedding_tensor: imgs,
                labels: lbls
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

        fpr, tpr, threshold = roc_curve(label_list, dist_list, pos_label=1)
        fnr = 1 - tpr
        # eer_threshold = threshold(np.nanargmin(np.absolute((fnr - fpr))))
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        print(eer)

        ### distance plot
        in_dist = []
        out_dist = []
        for i in range(len(label_list)):
            if label_list[i] == 0:
                out_dist.append(dist_list[i])
            else:
                in_dist.append(dist_list[i])

        in_dist = np.array(in_dist)
        out_dist = np.array(out_dist)

        plt.clf()
        plt.hist(in_dist, bins=100, histtype='bar', color='r', alpha=0.5, density=True, label='in_dist')
        plt.hist(out_dist, bins=100, histtype='bar', color='g', alpha=0.5, density=True, label='out_dist_face')

        plt.xlabel('Distance')
        plt.title('Histogram')
        plt.legend(loc='upper right')

        if is_permute:
            plt.savefig('./dist_histogram.png')
        else:
            plt.savefig('./dist_histogram_original.png')

        mated_list = []

        for i, lbl in enumerate(label_list):
            if lbl == 1:
                mated_list.append(dist_list[i])

        np.savetxt('./unlinkability/genuine.txt', mated_list)

        # ### Get gallery hash code
        # gallery = []
        # gallery_label = []
        # sess.run(iterator['gallery'].initializer)
        # try:
        #     while True:
        #         img, lbl = sess.run([image['gallery'], label['gallery']])
        #
        #         gallery_dict = {
        #             images: img,
        #             train_phase_dropout: False,
        #             train_phase_bn: False
        #         }
        #
        #         hash_code = sess.run(h_norm, feed_dict=gallery_dict)
        #
        #         if gallery == []:
        #             gallery = hash_code
        #             gallery_label = lbl
        #         else:
        #             gallery = np.concatenate((gallery, hash_code), axis=0)
        #             gallery_label = np.concatenate((gallery_label, lbl), axis=0)
        #
        # except tf.errors.OutOfRangeError:
        #     pass
        #
        # ### Get probe hash code
        # probe = []
        # probe_label = []
        # code_arr = []
        # sess.run(iterator['test'].initializer)
        # try:
        #     while True:
        #         img, lbl = sess.run([image['test'], label['test']])
        #
        #         gallery_dict = {
        #             images: img,
        #             train_phase_dropout: False,
        #             train_phase_bn: False
        #         }
        #
        #         code, hash_code = sess.run([h, h_norm], feed_dict=gallery_dict)
        #
        #         if probe == []:
        #             probe = hash_code
        #             probe_label = lbl
        #             code_arr = code
        #         else:
        #             probe = np.concatenate((probe, hash_code), axis=0)
        #             probe_label = np.concatenate((probe_label, lbl), axis=0)
        #             code_arr = np.concatenate((code_arr, code), axis=0)
        #
        # except tf.errors.OutOfRangeError:
        #     pass
        #
        # ### Calculate MAP
        # gtp = 40
        # k = 50
        #
        # distance = np.matmul(probe, gallery.T)
        # arg_idx = np.argsort(-distance, axis=1)
        #
        # max_label = gallery_label[arg_idx[:, :k]]
        # match_matrix = np.equal(max_label, probe_label[:, np.newaxis])
        #
        # tp_seen = match_matrix * np.cumsum(match_matrix, axis=1)
        # ap = np.sum(tp_seen / np.arange(1, k + 1)[np.newaxis, :], axis=1) / gtp
        # MAP = np.mean(ap)
        #
        # print(MAP)