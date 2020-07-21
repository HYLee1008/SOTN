import tensorflow as tf
import numpy as np
import time
import os
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


from src.model import get_args
from src.funcs import linear
from src.youtubeface import load_ytf_data
from src.lfw import load_lfw_data
from src.facescrub import load_fs_data
from src.wrapper_basicImg import wrapper_basicImg




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    total_iteration = 300000
    m = 512
    q = 32
    lam = 0.01
    beta = 1.
    margin = 0.5
    s = 32
    batch_size = 256
    class_num = 1595
    train_dataset = 'FS'
    eval_dataset = "LFW"
    args = get_args()

    ### Get image and label from tfrecord
    image, label, iterator = {}, {}, {}
    if train_dataset == 'YTF':
        image['train'], label['train'], iterator['train'] = load_ytf_data(batch_size, 'train')

    elif train_dataset == 'FS':
        image['train'], label['train'], iterator['train'] = load_fs_data(batch_size, 'train')

    else:
        print("Select proper dataset")

    ### Get evaluation dataset. Wrapper
    wrapper = wrapper_basicImg(dataset=eval_dataset)
    if eval_dataset == 'YTF':
        image['gallery'], label['gallery'], iterator['gallery'] = load_ytf_data(batch_size, 'train', eval=True)
        image['test'], label['test'], iterator['test'] = load_ytf_data(batch_size, 'test')

    elif eval_dataset == 'LFW':
        image['gallery'], label['gallery'], iterator['gallery'] = load_lfw_data(batch_size, 'gallery')
        image['test'], label['test'], iterator['test'] = load_lfw_data(batch_size, 'probe')


    ### Backbone network (Arcface)
    embedding_tensor = tf.placeholder(name='img_inputs', shape=[None, 512], dtype=tf.float32)
    labels = tf.placeholder(name='label', shape=[None, ], dtype=tf.int32)

    ### Global step & learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.003
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, total_iteration, 0.96)

    ### My implementation (DIom algorithm)
    with tf.variable_scope('DIom'):
        fc1 = linear(tf.nn.relu(embedding_tensor), 1024, 'fc1')
        fc2 = linear(tf.nn.relu(fc1), 1024, 'fc2')
        fc3 = linear(tf.nn.relu(fc2), m * q, 'fc3')

        h_k = tf.reshape(fc3, [-1, m, q])
        h_k = tf.nn.softmax(beta * h_k, axis=2)

        index_matrix = tf.range(1, q + 1, dtype=tf.float32)
        h = tf.reduce_sum(h_k * index_matrix, axis=2)
        h = tf.reshape(h, [-1,  m])
        h_norm = tf.math.l2_normalize(h, axis=1)

    ### Loss function
    l = tf.one_hot(labels, class_num)
    l = tf.matmul(l, tf.transpose(l))
    l_float = tf.cast(l, tf.float32)
    l = tf.reshape(tf.clip_by_value(l_float, 0., 1.), (-1, 1))
    label_int = tf.cast(tf.squeeze(l, 1), tf.int32)

    inner_prod = tf.reshape(tf.matmul(h_norm, tf.transpose(h_norm)), (-1, 1))
    cos_t = tf.clip_by_value(inner_prod, -1., 1. - 1e-6)
    theta = tf.math.acos(cos_t)

    sin_t = tf.math.sin(theta)
    cos_mt = tf.math.cos(theta + margin)
    sin_mt = tf.math.sin(theta + margin)

    logit = l * s * (tf.concat([sin_t, cos_mt], 1)) + (1 - l) * s * (tf.concat([sin_mt, cos_t], 1))

    l_ij_logit = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label_int)
    c_ij = tf.abs(tf.reduce_mean(h, axis=0) - (q + 1) / 2)

    # Baseline pairwise-CE
    # label_ce = tf.cast(labels, tf.float32)
    # l_ij = l * tf.log(tf.square(inner_prod)) + (1 - l) * tf.log(tf.maximum(1e-6, 1 - tf.square(inner_prod)))
    # l_ij = -tf.reduce_mean(l_ij)

    # My novel cosine loss
    l_ij = tf.reduce_mean(l_ij_logit)
    c_ij = tf.reduce_mean(c_ij)

    loss = l_ij + lam * c_ij

    gradient = tf.gradients(loss, sin_t)

    ### Optimizer
    t_vars = tf.global_variables()
    train_vars = [var for var in t_vars if 'DIom' in var.name]


    opt_t = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss, var_list=train_vars, global_step=global_step)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(iterator['train'].initializer)

        ### Training
        iteration = sess.run(global_step)
        t_opt = [opt_t, loss, l_ij, c_ij]
        start_time = time.time()
        while iteration != total_iteration:
            img, lbl = sess.run([image['train'], label['train']])

            train_dict = {
                embedding_tensor: img,
                labels: lbl
            }

            _, train_loss, loss_l, loss_c  = sess.run(t_opt, feed_dict=train_dict)
            iteration += 1

            if iteration % 10000 == 0:
                ### Evaluation after training
                ### Get gallery hash code
                # gallery = []
                # gallery_label = []
                # sess.run(iterator['gallery'].initializer)
                # try:
                #     while True:
                #         img, lbl = sess.run([image['gallery'], label['gallery']])
                #
                #         gallery_dict = {
                #             embedding_tensor: img
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
                #             embedding_tensor: img
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
                # ### Code frequency
                # code_arr = np.around(code_arr)
                # count_arr = []
                # for i in range(q):
                #     count_arr.append(np.count_nonzero(code_arr == i + 1))
                #
                # plt.clf()
                # plt.bar(range(1, q+1), count_arr)
                # plt.savefig('./plt/code_' + str(iteration) + '.png')

                # ### Calculate MAP
                # gtp = 40
                # k = 50
                #
                # distance = np.matmul(probe, gallery.T)
                # arg_idx = np.argsort(-distance, axis=1)
                #
                # max_label = gallery_label[arg_idx[:, :k]]
                # match_matrix = np.equal(max_label, probe_label[:,np.newaxis])
                #
                # tp_seen = match_matrix * np.cumsum(match_matrix, axis=1)
                # ap = np.sum(tp_seen / np.arange(1, k + 1)[np.newaxis, :], axis=1) / gtp
                # MAP = np.mean(ap)

                ### Calculate EER
                dist_list = []
                label_list = []
                code_list = []
                while wrapper.samples_left > 0:
                    imgs, lbls = wrapper.get_next_batch(100)

                    imgs = np.reshape(imgs, [-1, 512])

                    eer_dict = {
                        embedding_tensor: imgs
                    }

                    code, int_code = sess.run([h_norm, h], feed_dict=eer_dict)
                    code = np.reshape(code, [-1, 2, m])

                    distance = np.sum(np.prod(code, axis=1), axis=1)

                    if dist_list == []:
                        dist_list = distance
                        label_list = lbls
                        code_list = int_code

                    else:
                        dist_list = np.concatenate((dist_list, distance), axis=0)
                        label_list = np.concatenate((label_list, lbls), axis=0)
                        code_list = np.concatenate((code_list, int_code), axis=0)

                wrapper.samples_left= np.size(wrapper.labels, axis=0)
                wrapper.next_batch_pointer = 0

                fpr, tpr, threshold = roc_curve(label_list, dist_list, pos_label=1)
                fnr = 1 - tpr
                # eer_threshold = threshold(np.nanargmin(np.absolute((fnr - fpr))))
                eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

                ### Code frequency
                code_arr = np.around(code_list)
                count_arr = []
                for i in range(q):
                    count_arr.append(np.count_nonzero(code_arr == i + 1))

                plt.clf()
                plt.bar(range(1, q + 1), count_arr)
                plt.savefig('./plt/code_' + str(iteration) + '.png')

                time_taken = time.time() - start_time
                MAP = 0
                # print("good")
                print("[Iteration %d] Train Loss: %.4f, Loss_l: %.4f, Loss_c: %.4f, MAP: %.4f, EER: %.4f, Taken time: %.4f"
                      % (iteration, train_loss, loss_l, loss_c, MAP, eer, time_taken))

                start_time = time.time()

        # np.save('CP.npy', np.concatenate((fpr[np.newaxis, :], tpr[np.newaxis, :]), axis=0))
        ### Save model.
        # save_vars = [var for var in t_vars if 'DIom' in var.name]
        # saver = tf.train.Saver(var_list=save_vars)
        # saver.save(sess, './model/DIom_layer')