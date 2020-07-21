import cv2
import numpy as np
import os
import glob
import tensorflow as tf
from scipy.misc import imread, imshow
import csv
import random
import yaml

from src.model import get_embd, get_args



def crop(filename, img_size):
    img = imread(filename)
    h = img.shape[0]
    w = img.shape[1]

    top = np.floor(0.5 * (h - h / 2.5)).astype(int)
    bottom = np.ceil(0.5 * (h + h / 2.5)).astype(int)
    left = np.floor(0.5 * (w - w / 2.5)).astype(int)
    right = np.ceil(0.5 * (w + w / 2.5)).astype(int)
    img_cropped = img[top:bottom, left:right, :]

    return cv2.resize(img_cropped, (img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)


def generate_croped_images():
    rootpath = '/home/hy/Dataset/face/Youtube Face/'
    readname = 'aligned_images_DB/'
    lst_name = os.listdir(rootpath + readname)
    savename = 'detected_faces/'

    for personname in lst_name:
        read_personfolder = rootpath + readname + personname
        lst_idx = os.listdir(read_personfolder)
        save_personfolder = rootpath + savename + personname
        try:
            os.mkdir(save_personfolder)
        except:
            pass

        for seqidx in lst_idx:
            save_seqfolder = save_personfolder + '/' + seqidx
            try:
                os.mkdir(save_seqfolder)
            except:
                pass

            for (i, img_file) in enumerate(glob.iglob(read_personfolder + '/' + seqidx + '/' + '*.jpg')):
                img_cropped = crop(img_file, [112, 112])
                savepath = save_seqfolder + '/cropped_{0:03d}.jpg'.format(i)
                cv2.imwrite(savepath, img_cropped)


def generate_tfrecord(sess, images, train_phase_dropout, train_phase_bn, embedding_tensor):
    # for tfrecord
    tfrecords_train = 'YTF_train.tfrecords'
    tfrecords_test = 'YTF_test.tfrecords'
    with tf.io.TFRecordWriter(tfrecords_train) as writer_train:
        with tf.io.TFRecordWriter(tfrecords_test) as writer_test:

            # Get images from directory
            rootpath = '/home/hy/Dataset/face/Youtube Face/'
            readname = 'detected_faces/'
            lst_name = os.listdir(rootpath + readname)

            for (l, personname) in enumerate(lst_name):
                read_personfolder = rootpath + readname + personname
                lst_idx = os.listdir(read_personfolder)
                i = 0

                for seqidx in lst_idx:
                    for img_file in glob.iglob(read_personfolder + '/' + seqidx + '/' + '*.jpg'):
                        if i < 40:
                            img = cv2.imread(img_file)
                            embd_dict = {
                                images: img,
                                train_phase_dropout: False,
                                train_phase_bn: False
                            }

                            embedding = sess.run(embedding_tensor, feed_dict=embd_dict)
                            embedding_raw = embedding.tostring()
                            name = personname.encode()
                            example = tf.train.Example(features=tf.train.Features(feature={
                                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[embedding_raw])),
                                'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[l]))
                            }))

                            writer_train.write(example.SerializeToString())
                            i += 1

                        elif i < 45:
                            img = cv2.imread(img_file)
                            embd_dict = {
                                images: img,
                                train_phase_dropout: False,
                                train_phase_bn: False
                            }

                            embedding = sess.run(embedding_tensor, feed_dict=embd_dict)
                            embedding_raw = embedding.tostring()
                            name = personname.encode()
                            example = tf.train.Example(features=tf.train.Features(feature={
                                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[embedding_raw])),
                                'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[l]))
                            }))

                            writer_test.write(example.SerializeToString())
                            i += 1

                        else:
                            i += 1
                            continue


def load_ytf_data(batch_size, fold, eval=False):
    def decode(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

        image = tf.decode_raw(features['image_raw'], out_type=tf.float32)
        image = tf.cast(tf.reshape(tensor=image, shape=[512]), tf.float32)
        label = tf.cast(features['label'], tf.int32)

        return image, label

    if fold == 'train':
        dataset = tf.data.TFRecordDataset(filenames='./src/YTF_train.tfrecords')
        dataset = dataset.map(lambda x : decode(x))
        if eval:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(10000).batch(batch_size).repeat()

        iterator = dataset.make_initializable_iterator()
        x, y = iterator.get_next()

        return x, y, iterator

    elif fold == 'test':
        dataset = tf.data.TFRecordDataset(filenames='./src/YTF_test.tfrecords')
        dataset = dataset.map(lambda x: decode(x))
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        x, y = iterator.get_next()

        return x, y, iterator


def loadLabelsFromRow(r, n):
    if (r[4] == '1'):
        return [1] * n
    else:
        return [0] * n

def loadImage(basedir, namenumber):
    directory = "{0}/{1}".format(basedir, namenumber)
    list = os.listdir(directory)  # dir is your directory path
    number_files = len(list)
    number = random.randint(0, number_files-1)

    filename = "{0}/{1}/{2}_{3:03d}.jpg".format(basedir, namenumber, 'cropped', int(number))
    return imread(filename)

def loadImagePairFromRow(basedir, r):
    return [loadImage(basedir, r[2]), loadImage(basedir, r[3])]


def load_images(basedir, rows, n):
    image_list = []

    for i, row in enumerate(rows):
        for _ in range(n):
            image_list.append(loadImagePairFromRow(basedir, row))

    return np.array(image_list)

def load_ytf_pairs(basedir, pair_txt, n):
    with open(pair_txt, 'r') as csvfile:
        trainrows = list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE, skipinitialspace=True))[1:]

    train_images = load_images(basedir, trainrows, n)
    train_labels = np.array(list(map(lambda r: loadLabelsFromRow(r, n), trainrows)))

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

    images = tf.roll(images, shift=1, axis=2)
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


if __name__ == '__main__':
    basedir = '/home/hy/Dataset/face/Youtube Face/detected_faces'
    pair_txt = '/home/hy/Dataset/face/Youtube Face/splits.txt'

    main()
    # a, _ = load_ytf_pairs(basedir, pair_txt, 1)

    # print("good")