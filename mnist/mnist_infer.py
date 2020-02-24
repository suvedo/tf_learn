# -*- coding = utf-8 -*-

import tensorflow as tf


input_layer_size  = 784
first_layer_size  = 128
secode_layer_size = 10
output_layer_size = 10 


def get_weight_variable(shape, regularizer = None):
    weight = tf.get_variable("weight", shape, 
        initializer = tf.truncated_normal_initializer(stddev = 0.1))
    if regularizer:
        tf.add_to_collection("losses", regularizer(weight))

    return weight


def infer(input_tensor, regularizer = None):
    with tf.variable_scope("first_layer") as scope1:
        weight1 = get_weight_variable([input_layer_size, first_layer_size],
            regularizer)
        bias1 = tf.get_variable("bias1", [first_layer_size],
            initializer = tf.constant_initializer(0.0))
        first_layer_output = tf.nn.relu(tf.matmul(input_tensor, weight1) + bias1)

    with tf.variable_scope("second_layer") as scope2:
        weight2 = get_weight_variable([first_layer_size, secode_layer_size],
            regularizer)
        bias2 = tf.get_variable("bias2", [secode_layer_size],
            initializer = tf.constant_initializer(0.0))
        secode_layer_output = tf.matmul(first_layer_output, weight2) + bias2

    return secode_layer_output

