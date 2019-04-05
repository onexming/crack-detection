import tensorflow as tf


class dataset():
    def __init__(self, data_dir, shuffle=True, batch_size=32):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.batch_size = batch_size

    def parse(self, example):
        parse_dict = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'piex': tf.FixedLenFeature([], tf.int64)
        }
        example = tf.parse_single_example(example, parse_dict)
        example['image'] = tf.decode_raw(example['image'], tf.uint8)
        example['image'] = tf.cast(example['image'], tf.float32)
        example['image'] = tf.reshape(example['image'], (example['piex'][0]), example['piex'][1], 3)
        if self.shuffle:
            example['image'] = tf.math.subtract(tf.math.divide(example, 127.5), 1.)
            example['image'] = tf.image.random_brightness(example['image'], 0.2)
            example['image'] = tf.image.random_contrast(example['image'], 0.5, 1.5)
