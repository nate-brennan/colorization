import tensorflow as tf

learning_rate = 10**-4


# I'm assuming we will resize all images to the same size
# tf.image.resize_images
img_h = ???
img_w = ???

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.zeros(shape))

def conv2d(input, filter):
  return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(input):
  return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

input = tf.placeholder(tf.int32, [None, img_h * img_w])
output = tf.placeholder(tf.int32, [None, img_h * img_w, 3])

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
input_reshape = tf.reshape(input, [-1,None,None,1])
h_conv1 = tf.nn.relu(conv2d(input_reshape, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
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
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])
result = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

loss = tf.sqrt(tf.reduce_sum(tf.square(tf.log(result + 1) - tf.log(output + 1)))/3)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)





