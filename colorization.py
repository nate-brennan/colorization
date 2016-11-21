import tensorflow as tf
import numpy as np
from IPython.display import Image, display

learning_rate = 1e-3

# I'm assuming we will resize all images to the same size
# tf.image.resize_images
img_h = 256
img_w = 256

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.zeros(shape))

def conv2d(input, filter):
  return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(input):
  return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def encode(tensor):
	return tf.image.encode_png(tf.cast(tensor, dtype=tf.uint8).eval()[0]).eval()

input = tf.placeholder(tf.float32, [None, img_h, img_w, 1])
output = tf.placeholder(tf.float32, [None, img_h, img_w, 3])

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
input_reshape = tf.reshape(input, [-1,img_h,img_w,1])
h_conv1 = tf.nn.relu(conv2d(input_reshape, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# first feed forward
W_fc1 = weight_variable([img_h/4 * img_w/4 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, img_h/4 * img_w/4 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second feed forward
W_fc2 = weight_variable([1024, img_h/4 * img_w/4 * 3])
b_fc2 = bias_variable([img_h/4 * img_w/4 * 3])
result_1D = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
result_small = tf.reshape(result_1D, [-1, img_h/4, img_w/4, 3])
result = tf.image.resize_images(result_small, img_h, img_w)

loss = tf.reduce_sum(tf.log(tf.abs(result - output) + 1))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	input_image = tf.image.decode_png(tf.read_file('biden-obama.png'), channels=1).eval()
	output_image = tf.image.decode_png(tf.read_file('biden-obama.png'), channels=3).eval()

	for i in range(10):
		train_loss, _ = sess.run([loss, train_step], feed_dict={input: [input_image], output: [output_image], keep_prob: .5})
		print(train_loss)

	test_loss, test_result = sess.run([loss,result], feed_dict={input: [input_image], output: [output_image], keep_prob: 1})
	print(test_loss)
	with open('result.png', 'w') as f:
		f.write(encode(test_result))




