import tensorflow as tf


def preprocess_module(input):
    # zero-mean input
    with tf.name_scope('preprocess') as scope:
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
                           shape=[1, 1, 1, 3], name='img_mean')
        images = input - mean
        return images


def cnn_module(name, input, kernel_shape, strides=1, padding='SAME', relu=True):
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal(kernel_shape,
                                                 dtype=tf.float32,
                                                 stddev=1e-1),
                             name=name + "_w")

        """
        For Padding = SAME
        out_height = ceil(float(in_height) / float(strides[1]))
        out_width  = ceil(float(in_width) / float(strides[2]))
        """

        conv = tf.nn.conv2d(input,
                            kernel,
                            [1, strides, strides, 1],
                            padding=padding)
        biases = tf.Variable(tf.constant(0.0,
                                         shape=[kernel_shape[-1]],
                                         dtype=tf.float32),
                             trainable=True,
                             name=name + "_b")

        out = tf.nn.bias_add(conv, biases)

        if relu:
            out = tf.nn.relu(out, name=scope)

        return out, [kernel, biases]


def max_pool_module(name, input, ksize=2, strides=2):
    pool = tf.nn.max_pool(input,
                          name=name,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, strides, strides, 1],
                          padding='SAME')
    return pool
