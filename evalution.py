# coding=utf-8
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
from model import model as Model


class evluation():
    def __init__(self, logdi, data_dir, model):
        assert model in ["test", "train"], "model error!"
        self.model = model
        self.file_list = glob.glob(os.path.join(data_dir, model + '*'))
        self.summary = logdi

    def data_input(self):
        file_queue = tf.train.string_input_producer(self.file_list)
        TFrecoder = tf.TFRecordReader()
        # read 的返回值有key , 和 value
        key, value = TFrecoder.read(file_queue)
        # print("key of .read is : ", key)
        raw_data = tf.parse_single_example(
            value,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64)
            })
        raw_data["image"] = tf.decode_raw(raw_data["image"], tf.uint8)
        raw_data["image"] = tf.cast(raw_data["image"], tf.float32)
        raw_data["image"] = tf.reshape(raw_data['image'], [227, 227, 3])
        raw_data["image"] = tf.image.per_image_standardization(raw_data["image"])
        data = tf.train.batch([raw_data['image'], raw_data['label']], 32, num_threads=2, capacity=128)
        return data

    def evaluation(self, model_dir, global_step):
        with tf.Graph().as_default() as g:
            image, label = self.data_input()
            prey = Model(image)
            prey = tf.argmax(prey, axis=1)
            accuracy, updata = tf.metrics.accuracy(label, prey)
            tf.summary.scalar(self.model + "_accuracy", accuracy)
            marge = tf.summary.merge_all()
            restore = tf.train.Saver()
            with tf.Session() as sess:
                # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                sess.run(tf.local_variables_initializer())
                coord = tf.train.Coordinator()
                thread = tf.train.start_queue_runners(sess, coord)
                restore.restore(sess, model_dir)
                for _ in range(100):
                    sess.run(updata)
                accuracy_val, marge_val = sess.run([accuracy, marge])
                self.summary.add_summary(marge_val, global_step)
                coord.request_stop()
                coord.join(thread)
        return accuracy_val

                # def evaluation(self, model_dir, global_step):

                #     with tf.Graph().as_default() as g:
                #         # data = self.data_input()
                #         X, Y = self.data_input()
                #         # print(X['image'])
                #         # print(X)
                #         # print(Y)
                #         summary = tf.summary.FileWriter(self.logdir, g)
                #         # 导入图的构造
                #         # print(self.model_dir)
                #         # print(model_dir)
                #         # parameter = tf.train.latest_checkpoint(model_dir)
                #         # print(parameter)
                #         saver = tf.train.import_meta_graph(model_dir + '.meta')
                #         input = g.get_tensor_by_name("IteratorGetNext:0")
                #         output = g.get_tensor_by_name("output/flatten/Reshape:0")
                #         # print(input)
                #         pre_Y = tf.argmax(output, axis=1)
                #
                #         accuracy, updata = tf.metrics.accuracy(Y, pre_Y)
                #         #
                #         # tf.summary.image("image", X)
                #         tf.summary.scalar(self.model + "_accuracy", accuracy)
                #         marge = tf.summary.merge_all()
                #         # writer = tf.summary.FileWriter("/home/wxming/PycharmProject/Notebook/Concrete-Crack-Detection/log/log/tarin"
                #         #                                , tf.get_default_graph())
                #         # writer.close()
                #
                #         with tf.Session() as sess:
                #             saver.restore(sess, model_dir)
                #             #此处不能使用global_variables_initializer，　否则导致ｒｅｓｔｏｒｅ失败
                #             #sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                #             # X = g.get_tensor_by_name("_make_dataset_KuDRUjN5G1w")
                #             # all_operation = g.get_operations()
                #             # all_operation = [var for var in all_operation if "IteratorGetNext" in var.name or  "flatten/Reshape" in var.name]
                #             # all_operation = [var for var in all_operation if "flatten/Reshape" in var.name]
                #             # sess.run([get_out], feed_dict={"IteratorGetNext:0":self.iter})
                #             # sess.run([get_out])
                #             # data, _ = sess.run(self.iter)
                #             # print(data)
                #             # print(self.X)
                #             # print(data)
                #             coor = tf.train.Coordinator()
                #             thread = tf.train.start_queue_runners(sess=sess, coord=coor)
                #             # for _ in range(3):
                #
                #             # print("pre_Y", sess.run(pre_Y))
                #             # print("label", sess.run(Y))
                #             # data = sess.run(X)
                #             # _, prey = sess.run([updata, pre_Y])
                #             # print("pre_y", prey)
                #             # print("accuracy", sess.run(accuracy))
                #             # print("accuracy", accuracy)
                #             print("---------")
                #             for _ in range(2):
                #
                #                 data, y = sess.run([X, Y])
                #                 # x, y = sess.run(data)
                #                 # print("prey", data)
                #                 # print("label", y)
                #                 a, _, prey = sess.run([accuracy, updata, pre_Y], feed_dict={input: data})
                #                 print("prey", a)
                #                 print("label    ", y)
                #
                #                 print("prey_true", prey)
                #
                #                 if _ == 9:
                #                     print("X=\n", len(data))
                #             data = sess.run(X)
                #             accuracy_val, marge_val = sess.run([accuracy, marge], feed_dict={input: data})
                #             summary.add_summary(marge_val, global_step=global_step)
                #             coor.request_stop()
                #             coor.join(thread)
                # return accuracy_val
                # sess.run([get_out], feed_dict={input: self.X})
                # print(get_out)
                # print(input)
                # print("all_operation:\n", all_operation)


# if __name__ == '__main__':
#     dir = "/home/wxming/PycharmProject/Notebook/Concrete-Crack-Detection/log/log/tarin/"
#     data_dir = "/home/wxming/PycharmProject/Data/TFdata/Concrete"
#
#     model_dir = "/home/wxming/PycharmProject/Notebook/Concrete-Crack-Detection/log/model"
#     parameter = tf.train.latest_checkpoint(model_dir)
#     print(parameter)
#     # print(parameter)
#     # a = tf.train.get_checkpoint_state(model).model_checkpoint_path
#     # print(a)
#
#     evalat = evluation(dir, data_dir, "test")
#     # evalat.test()
#     # accurcay = evalat.evaluation(parameter, 1)
#     accurcay = evalat.evaluation(
#         "/home/wxming/PycharmProject/Notebook/Concrete-Crack-Detection/log/model/model.cpk-61000", 1)
#     print(accurcay)
