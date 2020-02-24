import tensorflow as tf

a = tf.Variable(tf.constant([1], shape = [1]), name = 'a')
b = tf.Variable(tf.constant([2], shape = [1]), name = 'b')
c = tf.multiply(a, b)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "my_saved_model")
