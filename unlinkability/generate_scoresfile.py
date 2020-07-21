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


def generate_matrix(m, q):
    indices = np.zeros((m, q, 2))
    for i in range(m):
        index = np.arange(q)
        np.random.shuffle(index)
        index = np.stack((i * np.ones(q), index), axis=-1)

        indices[i] = index

    return indices


def permute(input, matrix):
    indices_tf = tf.convert_to_tensor(matrix, dtype=tf.int32)

    output = tf.transpose(input, (1, 2, 0))
    output = tf.gather_nd(output, indices_tf)
    output = tf.transpose(output, (2, 0, 1))

    return output


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
    wrapper = wrapper_basicImg('LFW')

    ### Backbone network (Arcface)
    images = tf.placeholder(name='img_inputs', shape=[None, *args.image_size], dtype=tf.float32)
    labels = tf.placeholder(name='label', shape=[None, ], dtype=tf.int32)
    random_matrix = tf.placeholder(name='random_matrix', shape=[m, q, 2], dtype=tf.int32)

    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
    config = yaml.load(open(args.config_path))

    image_ = images / 127.5 - 1
    inputs = tf.reshape(image_, [-1, *args.image_size])
    embedding_tensor, _ = get_embd(inputs, train_phase_dropout, train_phase_bn, config)

    ### My implementation (DIom algorithm)
    with tf.variable_scope('DIom'):
        fc1 = linear(tf.nn.relu(embedding_tensor), 1024, 'fc1')
        fc2 = linear(tf.nn.relu(fc1), 1024, 'fc2')
        fc3 = linear(tf.nn.relu(fc2), m * q, 'fc3')

        h_k_original = tf.reshape(fc3, [-1, m, q])
        ### permute
        h_k_permuted = permute(h_k_original, random_matrix)
        h_k = tf.nn.softmax(beta * h_k_permuted, axis=2)

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
        restore_network_vars = [var for var in t_vars if 'embd_extractor' in var.name]
        restore_DIom_vars = [var for var in t_vars if 'DIom' in var.name]

        saver = tf.train.Saver(var_list=restore_network_vars)
        saver.restore(sess, args.model_path)

        saver = tf.train.Saver(var_list=restore_DIom_vars)
        saver.restore(sess, './model/DIom_layer')

        ### Get protected codes (total 2 independent code at each identity)
        true_list = [True, False]
        for i in range(2):
            code_list = []
            label_list = []
            matrix = generate_matrix(m, q)

            while wrapper.samples_left > 0:
                imgs, lbls = wrapper.get_next_batch(100)

                eer_dict = {
                    images: imgs[:, i],
                    random_matrix: matrix,
                    train_phase_dropout: False,
                    train_phase_bn: False
                }

                code, tmp1, tmp2 = sess.run([h_norm, h_k_original, h_k_permuted], feed_dict=eer_dict)

                if code_list == []:
                    code_list = code
                    label_list = lbls
                else:
                    code_list = np.concatenate((code_list, code), axis=0)
                    label_list = np.concatenate((label_list, lbls), axis=0)

            wrapper.samples_left = np.size(wrapper.labels, axis=0)
            wrapper.next_batch_pointer = 0

            if i == 0:
                total_code = code_list
                total_label = label_list
            if i == 1:
                total_code = np.concatenate((total_code[:, np.newaxis, :], code_list[:, np.newaxis, :]), axis=1)

        # total_code = np.round(total_code)
        # total_code = total_code / np.linalg.norm(total_code, axis=2)[:, :, np.newaxis]
        score_list = np.sum(np.prod(total_code, axis=1), axis=1)

        fpr, tpr, threshold = roc_curve(label_list, score_list, pos_label=1)
        fnr = 1 - tpr
        # eer_threshold = threshold(np.nanargmin(np.absolute((fnr - fpr))))
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        print(eer)

        mated_list = []
        nonmated_list = []

        for i, lbl in enumerate(label_list):
            if lbl == 1:
                mated_list.append(score_list[i])
            elif lbl == 0:
                nonmated_list.append(score_list[i])

        np.savetxt('./unlinkability/mated.txt', mated_list)
        np.savetxt('./unlinkability/nonmated.txt', nonmated_list)