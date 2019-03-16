import tensorflow as tf

# Create some variables.
v1 = tf.get_variable("vgg_16/conv1/conv1_1/biases", shape=[64])
# v2 = tf.get_variable("v2", shape=[5])

with tf.Session() as sess:
    saver = tf.train.Saver()
    # Restore variables from disk.
    saver.restore(sess, "./data/imagenet_weights/vgg_16.ckpt")
    print(v1.eval())
