# coding=utf-8
import tensorflow as tf
from tensorflow.python import keras as K
from tensorflow.python.keras import layers


def model(is_train, drop_rate):
    # input : 227x227x3
    m = K.Sequential(name="crack_detection")
    m.add(layers.Conv2D(24, (20, 20), strides=2, padding="same", activation='relu'))
    m.add(layers.MaxPooling2D((7, 7), strides=2, padding="same"))
    m.add(layers.BatchNormalization(trainable=is_train))

    m.add(layers.Conv2D(48, (15, 15), strides=2, padding="valid", activation='relu'))
    m.add(layers.MaxPooling2D((4, 4), strides=2, padding="valid"))
    m.add(layers.BatchNormalization(trainable=is_train))

    m.add(layers.Conv2D(96, (10, 10), strides=2, padding="valid", activation='relu'))
    m.add(layers.Flatten())

    m.add(layers.Dropout(drop_rate))
    m.add(layers.Dense(2))
    return m


if __name__ == '__main__':
    m = model(True, 1)
    m.build((None, 227, 227, 3))
    print(m.summary())
    print(m.get_config())
