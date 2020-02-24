import tensorflow as tf
import numpy as np

c = np.random.random([4, 4])
b = tf.nn.embedding_lookup(c, (1, 2), (2, 3))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print (sess.run(b))
    print (c)
