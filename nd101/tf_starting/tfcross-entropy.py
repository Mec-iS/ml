"""
Cross-entropy:

    D(y^, y) = - SUM [y times log(y^)]

Kullback-Leibler (KL) divergence
"""

import tensorflow as tf 

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session

kl = - tf.reduce_sum(
    one_hot * tf.log(softmax) 
)

with tf.Session() as session:
    output = session.run(kl, feed_dict={softmax: softmax_data, one_hot: one_hot_data})
    print(output)