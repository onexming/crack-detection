# crtl + -/+ 折叠 +shift全部折叠
from core.model import model1 as model
import tensorflow as tf


# 抽象除自己的分类类, 返回结果0，1
class crack_detection():
    def __init__(self, model_dir):
        self.input = tf.placeholder(tf.float32, (227, 227, 3))
        ins = tf.image.per_image_standardization(self.input)
        ins = tf.expand_dims(ins, axis=0)

        # 模型
        self.out = tf.squeeze(tf.argmax(model(ins), axis=1))

        #  恢复模型
        saver = tf.train.Saver()

        # with tf.Session() as self.sess:
        self.sess = tf.Session()
        saver.restore(self.sess, model_dir)

    def run(self, da):
        return self.sess.run(self.out, feed_dict={self.input: da})


def slice_image_y(image, stride, image_size):
    h, _, _ = image.shape
    image_list = [[], []]
    for i in range(0, h, stride):
        if i + image_size > h:
            x = -image_size
            y = None
        else:
            x = i
            y = i + image_size
        im = image[x:y, :, :]
        image_list[0].append(im)
        image_list[1].append((x, y))
    return image_list


# 切分为一列的长条图片, 同时返回在图中的坐标
def slice_image_x(image, stride, image_size):
    _, w, _ = image.shape
    image_list = [[], []]
    for i in range(0, w, stride):
        if i + image_size > w:
            x = -image_size
            y = None
        else:
            x = i
            y = i + image_size
        im = image[:, x:y, :]
        image_list[0].append(im)
        image_list[1].append((x, y))
    return image_list


def slice_image(image, stride, image_size):
    slic_y = slice_image_y(image, stride, image_size)
    for ims, y in zip(slic_y[0], slic_y[1]):
        slic_x = slice_image_x(ims, stride, image_size)
        for im, x in zip(slic_x[0], slic_x[1]):
            yield im, y, x
