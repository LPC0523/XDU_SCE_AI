import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import random

input = 784                     #输入节点
output = 10                     #输出节点
Layer = 500                     #隐藏层节点
batch = 100                     #一批训练量
learning_rate_base = 0.3        #基础学习率
learning_rate_decay = 0.99      #学习衰减率
regularization_rate = 0.0001    #正则化系数
steps = 2000                    #训练轮数
moving_average_decay = 0.99     #滑动平均衰减率

MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"



def inference(input_tensor, avg_class, weight1, biase1, weight2, biase2):
    if (avg_class == None):
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biase1)
        return tf.matmul(layer1, weight2) + biase2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)) + biase1)
        return tf.matmul(layer1, avg_class.average(weight2)) + biase2


# 将 numpy 数组中的图片和标签顺序打乱
def shuffer_images_and_labels(images, labels):
    shuffle_indices = np.random.permutation(np.arange(len(images)))
    shuffled_images = images[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]
    return shuffled_images, shuffled_labels


# 将label从长度10的one hot向量转换为0~9的数字
# 例：get_label(total_labels[0]) 获取到total_labels中第一个标签对应的数字
def get_label(label):
    return np.argmax(label)


# images：训练集的feature部分
# labels：训练集的label部分
# batch_size： 每次训练的batch大小
# epoch_num： 训练的epochs数
# shuffle： 是否打乱数据
# 使用示例：
#   for (batchImages, batchLabels) in batch_iter(images_train, labels_train, batch_size, epoch_num, shuffle=True):
#       sess.run(feed_dict={inputLayer: batchImages, outputLabel: batchLabels})
def batch_iter(images, labels, batch_size, epoch_num, shuffle=True):
    data_size = len(images)

    num_batches_per_epoch = int(data_size / batch_size)  # 样本数/batch块大小,多出来的“尾数”，不要了

    for epoch in range(epoch_num):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))

            shuffled_data_feature = images[shuffle_indices]
            shuffled_data_label = labels[shuffle_indices]
        else:
            shuffled_data_feature = images
            shuffled_data_label = labels

        for batch_num in range(num_batches_per_epoch):  # batch_num取值0到num_batches_per_epoch-1
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield (shuffled_data_feature[start_index:end_index], shuffled_data_label[start_index:end_index])


# 读取数据集

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
def main(argv=None):
    train_and_test(mnist)


total_images = mnist.train.images
total_labels = mnist.train.labels
total_images, total_labels = shuffer_images_and_labels(total_images, total_labels)

# 简单划分前50000个为训练集，后5000个为测试集
origin_images_train = total_images[:50000]
origin_labels_train = total_labels[:50000]
origin_images_test = total_images[50000:]
origin_labels_test = total_labels[50000:]


# 构建和训练模型
def train_and_test(mnist):
    x = tf.placeholder(tf.float32, [None, input], name='x')
    y = tf.placeholder(tf.float32, [None, output], name='y')
    weight1 = tf.Variable(tf.truncated_normal([input, Layer], stddev=0.1))
    biase1 = tf.Variable(tf.constant(0.1, shape=[Layer]))

    weight2 = tf.Variable(tf.truncated_normal([Layer, output], stddev=0.1))
    biase2 = tf.Variable(tf.constant(0.1, shape=[output]))

    y1 = inference(x, None, weight1, biase1, weight2, biase2)
    step1= tf.Variable(0, trainable=False)
    varible_averages = tf.train.ExponentialMovingAverage(moving_average_decay, step1)
    varible_averages_opposite = varible_averages.apply(tf.trainable_variables())
    y_average = inference(x, varible_averages, weight1, biase1, weight2, biase2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y1, labels=tf.argmax(y, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizar = tf.contrib.layers.l2_regularizer(regularization_rate)
    regularization = regularizar(weight1) + regularizar(weight2)
    loss = cross_entropy_mean + regularization

    learning_rate1 = tf.train.exponential_decay(learning_rate_base, step1, mnist.train.num_examples / batch,decay_rate=learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate1).minimize(loss, step1)

    train_op = tf.group()
    with tf.control_dependencies([train_step, varible_averages_opposite]):
        train_op = tf.no_op(name='trian_and_test')
    correct_prediction = tf.equal(tf.argmax(y_average, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y: mnist.test.labels}
        for i in range(steps):
            if (i % 100 == 0):
                validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy "
                      "using average model is %g " % (i, validate_accuracy))
            xs, ys = mnist.train.next_batch(batch)
            sess.run(train_op, feed_dict={x: xs, y: ys})

        test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training steps, test accuracy using average "
          "model is %g" % (steps, test_accuracy))
    pass

    saver = tf.train.Saver()
    with tf.Session() as sess:                  #声明一个会话

        tf.global_variables_initializer().run()

        for i in range(steps):
            xs, ys = mnist.train.next_batch(batch)
            _, loss_value, step = sess.run([train_op, loss, step1], feed_dict={x: xs, y: ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step1)

# 划分数据集并调用train_and_test测试和验证
def hold_out(images, labels, train_percentage):
    pass



def cross_validation(images, labels, k):
    pass


# 使用简单划分的训练集和测试集训练，并使用测试集评估模型
# train_and_test(origin_images_train, origin_labels_train, origin_images_test, origin_labels_test, origin_images_test, origin_labels_test)

# 调用函数用留出法和k折交叉验证法评估模型
# hold_out(total_images, total_labels, 0.8)
# cross_validation(total_images, total_labels, 10)


if __name__ == '__main__':
    tf.app.run()