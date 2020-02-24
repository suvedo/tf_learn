# -*- coding = utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_infer


batch_size = 32
regularizer_lambda = 0.001
moving_average_decay = 0.999
learning_rate_base = 0.1
learning_rate_decay = 0.99
train_step_num = 100


def train(mnist_data):
    x  = tf.placeholder(tf.float32, [None, mnist_infer.input_layer_size], name = 'x')
    y_ = tf.placeholder(tf.float32, [None, mnist_infer.output_layer_size], name = 'y_')

    regularizer = tf.contrib.layers.l2_regularizer(regularizer_lambda)
    y = mnist_infer.infer(x, regularizer)

    step = tf.Variable(0, dtype = tf.int32, shape = [], trainable = False)
    variable_average = tf.train.ExponentialMovingAverage(
        moving_average_decay, step)
    moving_average_op = variable_average.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = y, labels = tf.argmax(y_, 1))
    cross_mean = tf.reduce_mean(cross_entropy)
    loss = cross_mean + tf.add_n(tf.get_collection("losses"))

    lr = tf.train.exponential_decay(learning_rate_base, step, 
        mnist_data.train.num_examples / batch_size, learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step = step)

    with tf.control_dependencies([train_step, moving_average_op]):
        train_op = tf.no_op(name = "train")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(train_step_num):
            xs, ys = mnist_data.train.next_batch(batch_size)
            #print ("lh_debug", xs, ys)
            _, iloss, istep = sess.run([train_op, loss, step], 
                feed_dict = {x : xs, y_ : ys})
            print ("lh_debug:: step:", sess.run(step))


def main(argv = None):
    mnist_data = input_data.read_data_sets("./mnist_data/data", one_hot = True)
    train(mnist_data)

main()




