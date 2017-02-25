import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# using feed_dict to populate palceholder at run
x = tf.placeholder(tf.string)

with tf.Session() as session:
    output = session.run(x, feed_dict={x: 'Hello World!'})
    print(output)

del x

y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as session:
    output = session.run(y, feed_dict={y: 34, z: 2.3453})
    print(output)

del y
del z

# operations

x = tf.subtract(tf.constant(2),tf.constant(1))

with tf.Session() as session:
    output = session.run(x)
    print(output)

del x

# casting

x = tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))

with tf.Session() as session:
    output = session.run(x)
    print(output)

del x
# complex operation

x = 10
y = 2
node = tf.subtract(
    tf.divide(
        tf.cast(tf.constant(x), tf.float32), 
        tf.cast(tf.constant(y), tf.float32)
    ), 
    tf.cast(tf.constant(1), tf.float32)
)
with tf.Session() as session:
    output = session.run(node)
    print(output)