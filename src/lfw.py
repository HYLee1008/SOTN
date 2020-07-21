import csv
import tarfile
from scipy.misc import imread, imshow
import numpy as np
import gzip
import shutil
import tensorflow as tf
import yaml

from src.model import get_embd, get_args


def crop_and_downsample(originalX):
    """
    Starts with a 250 x 250 image.
    Crops to 128 x 128 around the center.
    Downsamples the image to (downsample_size) x (downsample_size).
    Returns an image with dimensions (channel, width, height).
    """
    current_dim = 250
    target_dim = 112
    margin = int((current_dim - target_dim)/2)
    left_margin = margin
    right_margin = current_dim - margin

    if originalX.ndim == 3:
        newim = originalX[left_margin:right_margin, left_margin:right_margin, :]

    elif originalX.ndim == 4:
        newim = originalX[:, left_margin:right_margin, left_margin:right_margin, :]

    # the next line is important.
    # if you don't normalize your data, all predictions will be 0 forever.

    return newim


def loadLabelsFromRow(r):
    if (len(r) == 3):
        return 1
    else:
        return 0

def loadImage(tar, basename, name, number):
    filename = "{0}/{1}/{1}_{2:04d}.jpg".format(basename, name, int(number))
    return imread(tar.extractfile(filename))

def loadImagePairFromRow(tar, basename, r):
    if (len(r) == 3):
        # same
        return [loadImage(tar, basename, r[0], r[1]), loadImage(tar, basename, r[0], r[2])]
    else:
        # different
        return [loadImage(tar, basename, r[0], r[1]), loadImage(tar, basename, r[2], r[3])]

def load_images(tar, basename, rows):
    image_list = []

    for i, row in enumerate(rows):
        image_list.append(loadImagePairFromRow(tar, basename, row))

    return np.array(image_list)

def load_lfw_pairs():
    basename = '/home/hy/Dataset/face/LFW/lfw-deepfunneled'
    pair_txt = '/home/hy/Dataset/face/LFW/pairs.txt'

    tgz_filename = "{}.tgz".format(basename)
    tar_filename = "{}.tar".format(basename)
    tar_subdir = "lfw-deepfunneled"

    # it will be faster to decompress this tar file all at once
    print("--> Converting {} to tar".format(tgz_filename))
    with gzip.open(tgz_filename, 'rb') as f_in, open(tar_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    tar = tarfile.open(tar_filename)

    with open(pair_txt, 'r') as csvfile:
        trainrows = list(csv.reader(csvfile, delimiter='\t'))[1:]

    train_images = load_images(tar, tar_subdir, trainrows)
    train_labels = np.array(list(map(lambda r: loadLabelsFromRow(r), trainrows)))

    train_images = np.asarray([crop_and_downsample(x) for x in train_images])

    # train_images = np.reshape(train_images, (10, 600, 2, 112, 112, 3))
    # train_labels = np.reshape(train_labels, (10, 600))
    train_images = np.reshape(train_images, (6000, 2, 112, 112, 3))
    train_labels = np.reshape(train_labels, (6000))

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

        train_embeddings = np.zeros((6000, 2, 512))

        for i, imgs in enumerate(train_images):
            embd_dict = {
                images: imgs,
                train_phase_dropout: False,
                train_phase_bn: False
            }

            embeddings = sess.run(embedding_tensor, feed_dict=embd_dict)
            train_embeddings[i] = embeddings

    return train_embeddings, train_labels


def generate_tfrecord():
    def write_image(img, writer):
        img_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))

        writer.write(example.SerializeToString())

    # for tfrecord
    tfrecords_gallery = 'LFW_gallery.tfrecords'
    tfrecords_probe = 'LFW_probe.tfrecords'
    with tf.io.TFRecordWriter(tfrecords_gallery) as writer_gallery:
        with tf.io.TFRecordWriter(tfrecords_probe) as writer_probe:
            # Get images from tar file
            filename = '/home/hy/Dataset/face/LFW/lfw-deepfunneled.tar'

            with tarfile.open(filename) as tar:
                identities = [member for member in tar.getmembers()]
                label = 0
                image_dict = {}
                for identity in identities:
                    dirname = identity.name
                    name = dirname.split('/')[1]

                    if name in image_dict:
                        image_dict[name].append(crop_and_downsample(imread(tar.extractfile(identity))))
                    else:
                        image_dict[name] = [crop_and_downsample(imread(tar.extractfile(identity)))]

                for key in image_dict.keys():
                    if len(image_dict[key]) > 3:
                        for i, img in enumerate(image_dict[key]):
                            if i < 3:
                                write_image(img, writer_gallery)
                            else:
                                write_image(img, writer_probe)


def load_lfw_data(batch_size, fold):
    def decode(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

        image = tf.decode_raw(features['image_raw'], out_type=tf.uint8)
        image = tf.cast(tf.reshape(tensor=image, shape=[112, 112, 3]), tf.float32)
        label = tf.cast(features['label'], tf.int32)

        return image, label

    if fold == 'gallery':
        dataset = tf.data.TFRecordDataset(filenames='./src/LFW_gallery.tfrecords')
        dataset = dataset.map(lambda x: decode(x))
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        x, y = iterator.get_next()

        return x, y, iterator

    elif fold == 'probe':
        dataset = tf.data.TFRecordDataset(filenames='./src/LFW_probe.tfrecords')
        dataset = dataset.map(lambda x: decode(x))
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        x, y = iterator.get_next()

        return x, y, iterator


def count():
    c = 0
    for fn in ['LFW_probe.tfrecords']:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1

    print(c)


if __name__ == '__main__':
    # generate_tfrecord()
    count()