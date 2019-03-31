# coding=utf-8
import tensorflow as tf


def model1(X, drop=1.0, train=False):  # input 227x227x3
    # 方便获取图中节点
    input = tf.identity(X, "input")
    train = tf.identity(train, "BN_train")
    drop = tf.identity(drop, "drop_rate")

    layer = tf.layers.conv2d(input, filters=24, kernel_size=[20, 20], strides=2, padding='same')
    layer = tf.layers.max_pooling2d(layer, [7, 7], strides=2, padding='same')
    layer = tf.layers.batch_normalization(layer, training=train)
    layer = tf.nn.relu(layer)

    layer = tf.layers.conv2d(layer, filters=48, kernel_size=[15, 15], strides=2, padding='valid')
    layer = tf.layers.max_pooling2d(layer, pool_size=[4, 4], strides=2, padding='valid')
    layer = tf.layers.batch_normalization(layer, training=train)
    layer = tf.nn.relu(layer)

    layer = tf.layers.conv2d(layer, filters=96, kernel_size=[10, 10], strides=2, padding='valid')
    layer = tf.layers.flatten(layer)
    layer = tf.nn.relu(layer)

    layer = tf.layers.dropout(layer, rate=drop)
    out = tf.layers.dense(layer, 2)

    out = tf.identity(out, "out")
    return out



def model2(X, drop=1.0):  # input 227x227x3

    with tf.variable_scope("C1", reuse=tf.AUTO_REUSE):  # 114x114x24
        layer = tf.layers.conv2d(X, filters=24, kernel_size=[20, 20], strides=2, padding='same')
        layer = tf.layers.max_pooling2d(layer, [7, 7], strides=2, padding='same')
        layer = tf.layers.batch_normalization(layer, momentum=0.9)
        layer = tf.nn.relu(layer)
        # print(layer)

    with tf.variable_scope("C2", reuse=tf.AUTO_REUSE):  # 22x22x48
        layer = tf.layers.conv2d(layer, filters=48, kernel_size=[15, 15], strides=2, padding='valid')
        # layer = tf.nn.relu(layer)
        # print(layer)

        # with tf.variable_scope("C3"):  # 10x10x48
        layer = tf.layers.max_pooling2d(layer, pool_size=[4, 4], strides=2, padding='valid')
        layer = tf.layers.batch_normalization(layer, momentum=0.9)
        layer = tf.nn.relu(layer)

        # print(layer)

    with tf.variable_scope("C3", reuse=tf.AUTO_REUSE):  # 1x1x96
        layer = tf.layers.conv2d(layer, filters=96, kernel_size=[10, 10], strides=2, padding='valid')
        layer = tf.layers.batch_normalization(layer, momentum=0.9)
        layer = tf.nn.relu(layer)

        # print(layer)
    with tf.variable_scope("drop_relu", reuse=tf.AUTO_REUSE):
        layer = tf.layers.dropout(layer, rate=drop)
        # layer = tf.layers.dropout(layer, rate=0.5)
        layer = tf.nn.relu(layer)
    with tf.variable_scope("C4", reuse=tf.AUTO_REUSE):
        layer = tf.layers.conv2d(layer, filters=2, kernel_size=[1, 1], strides=1, padding='valid')
    with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
        layer = tf.nn.softmax(layer)
        layer = tf.layers.flatten(layer)
        # layer = tf.Print(layer, [layer])
        # print(layer)
    return layer