# coding=utf-8
import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
import random
import sys
import matplotlib.pyplot as plt

DATA_DIR = './data/raw_image'

assert tf.gfile.Exists(DATA_DIR), "No path or file found!"


# 制作每个数据的label, 得到每张图片的完整路径
def create_image_dict(dir, name_list):
    # 获取目录下的所有的子目录
    data_list = []
    sub_dirs = [x[0] for x in tf.gfile.Walk(dir)]  # 获得所有的文件夹路径
    # 得到的第一个目录是根目录
    is_root_dir = True
    for sud_dir in sub_dirs:
        # 跳过第一个根目录
        if is_root_dir == True:
            is_root_dir = False
            continue

        # 获取当前目录下的所有图片文件
        extensions = ['jpg', 'jpeg']
        file_list = []
        for extension in extensions:
            file_glob = os.path.join(sud_dir, "*." + extension)
            im_list = glob.glob(file_glob)  # 匹配目录下所有的图片
            file_list.extend(im_list)
        # 如果file_list为空,跳过本次循环
        if not file_list: continue
        # print(len(file_list))

        # 获取目录名字
        dir_name = os.path.basename(sud_dir)
        label_name = dir_name.lower()
        '''获得一个类别目录下的所有图片的完整路径'''
        for i in range(len(name_list)):
            if label_name == name_list[i]:
                for file in file_list:
                    data = {
                        "label": i,
                        "image": file,
                    }
                    data_list.append(data)
    print("处理出的列表长度为%d" % (len(data_list)))
    return data_list


# 创建tfrecoder文件
def create_tfrecoder(data_list, dir=None, shuffle=True):
    # 如果没有给定 dir 则讲之赋值, 如果该目录下有文件,则将全部删除
    if dir == None:
        dir = "./data/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        for file in os.listdir(dir):
            if not os.path.isdir(os.path.join(dir, file)):
                os.remove(os.path.join(dir, file))
    if not tf.gfile.Exists(dir):
        tf.gfile.MakeDirs(dir)
    # 打乱列表的数据
    if shuffle:
        random.shuffle(data_list)
    # 获取图片的尺寸信息
    wide, heigh = Image.open(data_list[0]['image']).size
    writer = []
    tfdata_name = ['train', 'test']
    # 拥有分别储存训练集和验证集数据
    # 控制每个文件的写入大小
    num_example = []
    for i in tfdata_name:
        writer.append(tf.python_io.TFRecordWriter(os.path.join(dir, i + "_0000" + ".tfrecords")))
        # 用于计数test和train已经写入的数量
        num_example.append(0)
    court = 0
    for one_data in data_list:
        im = Image.open(one_data['image'])
        # 转换为二进制的数据格式, 其中tobytes和tostring是同一个功能
        im = im.tobytes()
        feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[im])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[one_data['label']])),
            "piex": tf.train.Feature(int64_list=tf.train.Int64List(value=[wide, heigh]))
        }
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        # 随机分到测试集的概率为0.1
        idx = 0 if random.random() > 0.1 else 1
        writer[idx].write(example.SerializeToString())
        num_example[idx] += 1
        if num_example[idx] % 5000 == 0:
            writer[idx].close()
            writer[idx] = tf.python_io.TFRecordWriter(dir + "/%s_%d.tfrecords" % (tfdata_name[idx], num_example[idx]))
        # print("%s 完成 %d 个" % (tfdata_name[idx], num_example[idx]))
        court += 1
        sys.stdout.write("\r>>  Converting image {}/{}  <<".format(court, len(data_list)))
        sys.stdout.flush()
    writer[0].close()
    writer[1].close()
    print("\n数据转化完成")


def iterator(dir, model):
    # 找到目录所有满足要求的文件
    assert os.path.getsize(dir), "文件为空,文件不存在"
    name_list = glob.glob(os.path.join(dir, model + "*"))

    def parse(example):
        parse_dict = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
        data = tf.io.parse_single_example(example, parse_dict)
        image = data['image']
        label = data['label']
        # 将转化为原来的图片的uint8的数据格式
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [227, 227, 3])
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = tf.image.per_image_standardization(image)
        return image, label

    data = tf.data.TFRecordDataset(name_list)
    data = data.map(parse)
    data = data.shuffle(1000)
    if model == "train":
        data = data.repeat(-1)
    else:
        data = data.repeat(1)
    data = data.batch(32).make_one_shot_iterator()
    next_data = data.get_next()
    return next_data


'''处理高分辨率图片'''


class slide():
    def __init__(self, image_dir, strides):
        self.image_dir = image_dir
        self.image_name = os.path.basename(image_dir).split(".")[0]
        self.strides = strides
        self.image_size = 227
        self.div = self.image_size - self.strides
        im = Image.open(self.image_dir)
        self.im_array = np.array(im)

    # 因为电脑太菜的原因只能先将图片大图片处理成模型规定的大小后,
    # 转化为tfrecoder文件格式
    def break_image(self, image_dir):
        self.tf_dir = image_dir + ".tfrecoder"

        if os.path.exists(self.tf_dir) == 0:
            print("目标文件的tfrecoder文件不存在, 正在制作tfrecoder文件")
            tf_name = tf.python_io.TFRecordWriter(self.tf_dir)
            h, w, _ = np.shape(self.im_array)
            # 数量
            h_num, w_num = (h - self.div) // self.strides, (w - self.div) // self.strides
            for i in range(0, h, self.strides):
                if i + self.image_size > h:
                    break
                for j in range(0, w, self.strides):
                    if j + self.image_size > w: break
                    sptle = self.im_array[i:(i + self.image_size), j:(j + self.image_size), :]
                    featurn = {
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sptle.tobytes()])),
                        "h": tf.train.Feature(int64_list=tf.train.Int64List(value=[h_num])),
                        "w": tf.train.Feature(int64_list=tf.train.Int64List(value=[w_num]))}
                    feature = tf.train.Features(feature=featurn)
                    features = tf.train.Example(features=feature).SerializeToString()
                    tf_name.write(features)
            tf_name.close()
        else:
            print(self.image_name, "的tfrecoder文件以存在")

        # 进入解析阶段
        def parse(example):
            parse_dict = {
                'image': tf.FixedLenFeature([], tf.string),
                'h': tf.FixedLenFeature([], tf.int64),
                'w': tf.FixedLenFeature([], tf.int64)
            }
            data = tf.parse_single_example(example, parse_dict)
            data['image'] = tf.decode_raw(data['image'], tf.uint8)
            data['image'] = tf.cast(data['image'], tf.float32)
            data['image'] = tf.reshape(data['image'], [self.image_size, self.image_size, 3])
            data['image'] = tf.image.per_image_standardization(data['image'])
            return data

        data = tf.data.TFRecordDataset(self.tf_dir)
        data = data.map(parse)
        data = data.batch(1)
        data = data.repeat(1)
        data_initializable = data.make_initializable_iterator()
        next = data_initializable.get_next()
        return next, data_initializable.initializer

    # 将识别好的图片重新融合为原来的图片
    def marge_image(self, array_bool):
        image = np.zeros_like(self.im_array)
        h, w = np.shape(array_bool)
        for i in range(h):
            for j in range(w):
                if array_bool[i, j]:
                    hhh = (i * self.strides)
                    www = (j * self.strides)
                    image[hhh:(hhh + self.image_size), www:(www + self.image_size), :] = self.im_array[
                                                                                         hhh:(hhh + self.image_size),
                                                                                         www:(www + self.image_size), :]
        return image, self.im_array


if __name__ == '__main__':
    name_list = ['negative', 'positive']
    list_dir = create_image_dict(DATA_DIR, name_list)
    create_tfrecoder(list_dir)
