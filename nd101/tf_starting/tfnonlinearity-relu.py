# Solution is available in the other "solution.py" tab
import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# example saver
save_file = './model_nonlin_relu.ckpt'
saver = tf.train.Saver()

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model
init = tf.global_variables_initializer() 
output_hidden_1 = tf.add(tf.matmul(features, weights[0]), biases[0])
output_hidden_2 = tf.nn.relu(output_hidden_1)
output_final = tf.add(tf.matmul(output_hidden_2, weights[1]), biases[1])

# TODO: Print session results
with tf.Session() as session:
    session.run(init)
    # restore: saver.restore(sess, save_file)
    # save: saver.save(sess, save_file)
    output = session.run(output_final)
    print(output)