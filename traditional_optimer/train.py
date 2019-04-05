# coding=utf-8

import tensorflow as tf
from core.model import model
from raw_to_tfrecoder import iterator
from datetime import datetime
import time
from traditional_optimer.evalution import evluation
import os

# 关闭系统警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
work_dir = "model2/"
data_dir = "input/"
learn_data = 1e-3
log_dir = "log2/"


def train():
    with tf.Graph().as_default():
        X, Y = iterator(data_dir, "train")
        logits = model(X, 0.5, True)
        loss = tf.losses.sparse_softmax_cross_entropy(Y, logits)
        # 创建或或者获取global_step
        step = tf.train.get_or_create_global_step()
        # 使用衰减的学习率
        learn_rate = tf.train.exponential_decay(learning_rate=learn_data, global_step=step, decay_steps=500,
                                                decay_rate=0.98, staircase=True)
        with tf.name_scope("optimizer"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss, step)

        saver = tf.train.Saver()
        writer_log = tf.summary.FileWriter(log_dir, tf.get_default_graph())
        losss = tf.summary.scalar("loss", loss)
        learn = tf.summary.scalar("learn_rata", learn_rate)
        summary_merga = tf.summary.merge([losss, learn])

        # 定义测试
        test_val = evluation(writer_log, data_dir=data_dir, model="test")

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            patince = 100
            best_accuracy = 0
            duration = 0
            while True:
                star = time.time()
                loss_val, step_val, _ = sess.run([loss, step, optimizer])
                duration += time.time() - star
                # 每一百次打印一次loss
                if (step_val + 1) % 100 != 0:
                    continue
                summary_val = sess.run(summary_merga)
                writer_log.add_summary(summary_val, global_step=step_val + 1)
                print(
                    ">>>>>nowtime:<{0}>,loss = {1:f}, step = {2:}, cost time : {3:.3f}<<<<<".format(datetime.now(),
                                                                                                    loss_val,
                                                                                                    step_val + 1,
                                                                                                    duration))
                duration = 0
                # 每1000次进行一次测试
                if (step_val + 1) % 1000 != 0:
                    continue
                save_dir = saver.save(sess, work_dir + "late.cpk")
                accuracy_test = test_val.evaluation(save_dir, step_val)
                print(
                    "After {0} trians,the test accuray=>{1:0.6f}<,best_accuracy>{2:0.6f}".format(
                        step_val + 1,
                        # accuracy_train,
                        accuracy_test,
                        best_accuracy))
                if accuracy_test > best_accuracy:
                    best_accuracy = accuracy_test
                    patince = 100
                    saver.save(sess, work_dir + "model.cpk", global_step=step_val + 1)
                    print("------>  save dir is ", work_dir)
                else:
                    patince -= 1
                print("------>  patience=", patince)
                if patince <= 0:
                    break
            print("------>  best accuracy :", best_accuracy)


if __name__ == '__main__':
    train()
