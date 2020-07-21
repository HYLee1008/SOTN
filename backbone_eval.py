import tensorflow as tf
import yaml
import argparse
import numpy as np
import os
from sklearn.metrics import roc_curve, auc

from src.model import get_embd
from src.youtubeface import load_data
from src.wrapper_basicImg import wrapper_basicImg


def get_args():
    parser = argparse.ArgumentParser(description='input information')
    parser.add_argument('--image_size', default=[112, 112, 3], help='the image size')
    parser.add_argument('--mode', type=str, default='build', help='model mode: build')
    parser.add_argument('--config_path', type=str, default='./configs/config_ms1m_100.yaml', help='config path, used when mode is build')
    parser.add_argument('--model_path', type=str, default='./model/best-m-1006000', help='model path')
    parser.add_argument('--val_data', type=str, default='', help='val data, a dict with key as data name, value as data path')
    parser.add_argument('--train_mode', type=int, default=0, help='whether set train phase to True when getting embds. zero means False, one means True')
    parser.add_argument('--target_far', type=float, default=1e-3, help='target far when calculate tar')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    total_iteration = 30000
    m = 512
    q = 32
    lam = 0.001
    beta = 9.
    margin = 0.5
    s = 32
    batch_size = 256
    class_num = 1595
    args = get_args()

    wrapper = wrapper_basicImg()

    ### Fold
    fold = tf.placeholder(tf.string)

    ### Get image and label from tfrecord
    image, label, iterator = {}, {}, {}
    image['train'], label['train'], iterator['train'] = load_data(batch_size, 'train')
    image['gallery'], label['gallery'], iterator['gallery'] = load_data(batch_size, 'train', eval=True)
    image['test'], label['test'], iterator['test'] = load_data(batch_size, 'test')

    ### Backbone network (Arcface)
    images = tf.placeholder(name='img_inputs', shape=[None, *args.image_size], dtype=tf.float32)
    labels = tf.placeholder(name='label', shape=[None, ], dtype=tf.int32)

    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
    config = yaml.load(open(args.config_path))

    image_ = images / 127.5 - 1
    # image_ = tf.roll(image_, shift=2, axis=3)
    inputs = tf.reshape(image_, [-1, *args.image_size])
    embedding_tensor, _ = get_embd(inputs, train_phase_dropout, train_phase_bn, config)

    norm = tf.math.l2_normalize(embedding_tensor, axis=1)

    ### Graph & load setup
    t_vars = tf.global_variables()
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    restore_vars = [var for var in t_vars if 'embd_extractor' in var.name]

    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(var_list=restore_vars)
        saver.restore(sess, args.model_path)

        ### Get gallery hash code
        gallery = []
        gallery_label = []
        sess.run(iterator['gallery'].initializer)
        try:
            while True:
                img, lbl = sess.run([image['gallery'], label['gallery']])

                gallery_dict = {
                    images: img,
                    train_phase_dropout: False,
                    train_phase_bn: False
                }

                hash_code = sess.run(norm, feed_dict=gallery_dict)

                if gallery == []:
                    gallery = hash_code
                    gallery_label = lbl
                else:
                    gallery = np.concatenate((gallery, hash_code), axis=0)
                    gallery_label = np.concatenate((gallery_label, lbl), axis=0)

        except tf.errors.OutOfRangeError:
            pass

        ### Get probe hash code
        probe = []
        probe_label = []
        sess.run(iterator['test'].initializer)
        try:
            while True:
                img, lbl = sess.run([image['test'], label['test']])

                gallery_dict = {
                    images: img,
                    train_phase_dropout: False,
                    train_phase_bn: False
                }

                hash_code = sess.run(norm, feed_dict=gallery_dict)

                if probe == []:
                    probe = hash_code
                    probe_label = lbl
                else:
                    probe = np.concatenate((probe, hash_code), axis=0)
                    probe_label = np.concatenate((probe_label, lbl), axis=0)

        except tf.errors.OutOfRangeError:
            pass

        ### Calculate MAP
        gtp = 40
        k =50

        distance = np.matmul(probe, gallery.T)
        arg_idx = np.argsort(-distance, axis=1)

        max_label = gallery_label[arg_idx[:, :k]]
        match_matrix = np.equal(max_label, probe_label[:,np.newaxis])

        tp_seen = match_matrix * np.cumsum(match_matrix, axis=1)
        ap = np.sum(tp_seen / np.arange(1, k + 1)[np.newaxis, :], axis=1) / gtp
        MAP = np.mean(ap)

        ### Calculate EER
        dist_list = []
        label_list = []
        while wrapper.samples_left > 0:
            imgs, lbls = wrapper.get_next_batch(100)

            imgs = np.reshape(imgs, [-1, 112, 112, 3])

            eer_dict = {
                images: imgs,
                train_phase_dropout: False,
                train_phase_bn: False
            }

            code = sess.run(norm, feed_dict=eer_dict)
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
        print('MAP: %.4f, EER: %.4f' %(MAP, eer))