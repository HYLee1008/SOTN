import tensorflow as tf
import yaml
import numpy as np
import os
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from src.model import get_embd, get_args
from src.youtubeface import load_ytf_data
from src.wrapper_basicImg import wrapper_basicImg




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    m = 32
    q = 32
    n = 5
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
    embedding_tensor = tf.placeholder(name='img_inputs', shape=[None, 512], dtype=tf.float32)
    labels = tf.placeholder(name='label', shape=[None, ], dtype=tf.int32)

    # Index-of-Max hasing
    # w_i = tf.random.normal((m, 512, q))
    w_i = tf.placeholder(name='w_i', shape=[m, 512, q], dtype=tf.float32)
    x_k = tf.einsum('ij,kjl->ikl', embedding_tensor, w_i)

    x_k = tf.math.argmax(x_k, axis=2)

    ### Graph & load setup
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    t_vars = tf.global_variables()
    restore_vars = [var for var in t_vars if 'embd_extractor' in var.name]

    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()
        sess.run(iterator['train'].initializer)

        ### Calculate EER
        eer = []
        for _ in range(n):
            random_matrix = np.random.randn(m, 512, q)
            dist_list = []
            label_list = []
            while wrapper.samples_left > 0:
                imgs, lbls = wrapper.get_next_batch(100)

                imgs = np.reshape(imgs, [-1, 512])

                eer_dict = {
                    embedding_tensor: imgs,
                    w_i: random_matrix,
                }

                code, tmp = sess.run([x_k, w_i], feed_dict=eer_dict)
                code = code / np.linalg.norm(code, axis=1)[:, np.newaxis]
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

            tmp1 = tmp

            fpr, tpr, threshold = roc_curve(label_list, dist_list, pos_label=1)
            fnr = 1 - tpr
            # eer_threshold = threshold(np.nanargmin(np.absolute((fnr - fpr))))
            a = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            eer.append(a)
            print(a)

        print("EER: %.4f" % np.mean(eer))

        ### distance plot
        # in_dist = []
        # out_dist = []
        # for i in range(len(label_list)):
        #     if label_list[i] == 0:
        #         out_dist.append(dist_list[i])
        #     else:
        #         in_dist.append(dist_list[i])
        #
        # in_dist = np.array(in_dist)
        # out_dist = np.array(out_dist)
        #
        # plt.clf()
        # plt.hist(in_dist, bins=100, histtype='bar', color='r', alpha=0.5, density=True, label='in_dist')
        # plt.hist(out_dist, bins=100, histtype='bar', color='g', alpha=0.5, density=True, label='out_dist_face')
        #
        # plt.xlabel('Distance')
        # plt.title('Histogram')
        # plt.legend(loc='upper right')
        #
        # plt.savefig('./dist_histogram.png')