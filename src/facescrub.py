import numpy as np
import tensorflow as tf
from scipy.misc import imread, imshow
import yaml
import os
import random
from src.model import get_embd, get_args


def generate_tfrecord(sess, images, train_phase_dropout, train_phase_bn, embedding_tensor):
    # for tfrecord
    tfrecords_training= 'FS_train.tfrecords'

    with tf.io.TFRecordWriter(tfrecords_training) as writer_train:
        # Get images from tar file
        rootpath = '/home/hy/Dataset/face/FaceScrub/'
        readname = 'facescrub_mtcnn_112/'
        lst_name = os.listdir(rootpath + readname)

        for (l, personname) in enumerate(lst_name):
            read_personfolder = rootpath + readname + personname
            lst_idx = os.listdir(read_personfolder)

            for filenum in lst_idx:
                img_file = rootpath + readname + personname + '/' + filenum
                img = imread(img_file)
                embd_dict = {
                    images: img,
                    train_phase_dropout: False,
                    train_phase_bn: False
                }

                embedding = sess.run(embedding_tensor, feed_dict=embd_dict)
                embedding_raw = embedding.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'embd_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[embedding_raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[l]))
                }))

                writer_train.write(example.SerializeToString())


def load_fs_data(batch_size, fold, eval=False):
    def decode(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
            'label': tf.FixedLenFeature([], tf.int64),
            'embd_raw': tf.FixedLenFeature([], tf.string)
        })

        embd = tf.decode_raw(features['embd_raw'], out_type=tf.float32)
        embd = tf.cast(tf.reshape(tensor=embd, shape=[512]), tf.float32)
        # image = tf.roll(image, shift=1, axis=2)
        label = tf.cast(features['label'], tf.int32)

        return embd, label

    if fold == 'train':
        dataset = tf.data.TFRecordDataset(filenames='./src/FS_train.tfrecords')
        dataset = dataset.map(lambda x : decode(x))
        if eval:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(10000).batch(batch_size).repeat()

        iterator = dataset.make_initializable_iterator()
        x, y = iterator.get_next()

        return x, y, iterator

    elif fold == 'test':
        dataset = tf.data.TFRecordDataset(filenames='./src/FS_test.tfrecords')
        dataset = dataset.map(lambda x: decode(x))
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        x, y = iterator.get_next()

        return x, y, iterator

def load_fs_pairs(num):
    def get_imgfile(name, file_name):
        img_file = name + '/' + file_name

        return imread(img_file)
    rootpath = '/home/hy/Dataset/face/FaceScrub/'
    readname = 'facescrub_mtcnn_112/'
    lst_name = os.listdir(rootpath + readname)

    match_list = []
    nonmatch_list = []

    for i in range(num):
        match_name, nonmatch_name = random.sample(lst_name, 2)

        read_matchfolder = rootpath + readname + match_name
        lst_idx = os.listdir(read_matchfolder)

        m1, m2, nm1 = random.sample(lst_idx, 3)

        match_list.append([get_imgfile(read_matchfolder, m1), get_imgfile(read_matchfolder, m2)])

        read_nonmatchfolder = rootpath + readname + nonmatch_name
        lst_idx = os.listdir(read_nonmatchfolder)

        nm2 = random.sample(lst_idx, 1)[0]

        nonmatch_list.append([get_imgfile(read_matchfolder, nm1), get_imgfile(read_nonmatchfolder, nm2)])

    total_list = np.concatenate((np.array(match_list), np.array(nonmatch_list)), axis=0)
    total_label = np.concatenate((np.ones(num), np.zeros(num)))

    # Graph
    ### Graph
    args = get_args()

    images = tf.placeholder(name='img_inputs', shape=[2, 112, 112, 3], dtype=tf.float32)

    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
    config = yaml.load(open('./configs/config_ms1m_100.yaml'))

    image_ = images / 127.5 - 1
    # image_ = tf.roll(image_, shift=2, axis=3)
    inputs = tf.reshape(image_, [-1, *args.image_size])
    embedding_tensor, _ = get_embd(inputs, train_phase_dropout, train_phase_bn, config)

    t_vars = tf.global_variables()
    restore_vars = [var for var in t_vars if 'embd_extractor' in var.name]

    ### Graph & load setup
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(var_list=restore_vars)
        saver.restore(sess, './model/best-m-1006000')

        train_embeddings = np.zeros((num * 2, 2, 512))

        for i, imgs in enumerate(total_list):
            embd_dict = {
                images: imgs,
                train_phase_dropout: False,
                train_phase_bn: False
            }

            embeddings = sess.run(embedding_tensor, feed_dict=embd_dict)
            train_embeddings[i] = embeddings

    return train_embeddings, total_label



    ### Graph
    args = get_args()

    images = tf.placeholder(name='img_inputs', shape=[2, 112, 112, 3], dtype=tf.float32)

    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
    config = yaml.load(open('./configs/config_ms1m_100.yaml'))

    images = tf.roll(images, shift=1, axis=3)
    image_ = images / 127.5 - 1
    # image_ = tf.roll(image_, shift=2, axis=3)
    inputs = tf.reshape(image_, [-1, *args.image_size])
    embedding_tensor, _ = get_embd(inputs, train_phase_dropout, train_phase_bn, config)

    t_vars = tf.global_variables()
    restore_vars = [var for var in t_vars if 'embd_extractor' in var.name]

    ### Graph & load setup
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(var_list=restore_vars)
        saver.restore(sess, './model/best-m-1006000')

        train_embeddings = np.zeros((5000, 2, 512))

        for i, imgs in enumerate(train_images):
            embd_dict = {
                images: imgs,
                train_phase_dropout: False,
                train_phase_bn: False
            }

            embeddings = sess.run(embedding_tensor, feed_dict=embd_dict)
            train_embeddings[i] = embeddings


    return train_embeddings, np.ravel(train_labels)


def main():
    ### Backbone network (Arcface)
    args = get_args()

    images = tf.placeholder(name='img_inputs', shape=[112, 112, 3], dtype=tf.float32)

    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
    config = yaml.load(open('../configs/config_ms1m_100.yaml'))

    image_ = images / 127.5 - 1
    # image_ = tf.roll(image_, shift=2, axis=3)
    inputs = tf.reshape(image_, [-1, *args.image_size])
    embedding_tensor, _ = get_embd(inputs, train_phase_dropout, train_phase_bn, config)

    t_vars = tf.global_variables()
    restore_vars = [var for var in t_vars if 'embd_extractor' in var.name]

    ### Graph & load setup
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(var_list=restore_vars)
        saver.restore(sess, '../model/best-m-1006000')

        generate_tfrecord(sess, images, train_phase_dropout, train_phase_bn, embedding_tensor)


if __name__ == "__main__":
    main()