import tensorflow as tf


x = tf.compat.v1.placeholder(tf.float32, name="x", shape=[None, 784])
W = tf.Variable(tf.random.uniform([784, 10], -1, 1), name="W")
b = tf.Variable(tf.zeros([10]), name="biases")
output = tf.matmul(x, W) + b
"""
weights = tf.Variable(
tf.random.normal([300, 200], stddev=0.5),
name="weights"
)

x = tf.compat.v1.placeholder(tf.float32, name="x", shape=[None, 784])
"""
