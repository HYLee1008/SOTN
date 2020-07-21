import tensorflow as tf
import numpy as np


def linear(input_, output_size, name, stddev=None,
           bias_start=0.0, with_biases=True,
           with_w=False):
    shape = input_.get_shape().as_list()

    ### 171210 I add this
    dim = np.prod(shape[1:])
    input_ = tf.reshape(input_, [-1, dim])
    shape = input_.get_shape().as_list()
    ###


    ### 180103 improved wgan github
    # https://github.com/igul222/improved_wgan_training/blob/master/tflib/ops/linear.py
    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')

    weight = uniform(
        np.sqrt(2. / (shape[1] + output_size)),
        (shape[1], output_size)
    )
    ###


    if stddev is None:
        stddev = np.sqrt(1. / (shape[1]))
    with tf.variable_scope(name) as scope:
        #if scope_has_variables(scope):
        #   scope.reuse_variables()
        weight = tf.get_variable("w", [shape[1], output_size], tf.float32,
                                tf.truncated_normal_initializer(stddev=stddev), regularizer=tf.contrib.layers.l2_regularizer(0.1))
        # weight = tf.get_variable("w", initializer=tf.constant(weight))
        if with_biases:
            bias = tf.get_variable("b", [output_size],
                                   initializer=tf.constant_initializer(bias_start))

        mul = tf.matmul(input_, weight)
        if with_w:
            if with_biases:
                return mul + bias, weight, bias
            else:
                return mul, weight, None
        else:
            if with_biases:
                return mul + bias
            else:
                return mul

