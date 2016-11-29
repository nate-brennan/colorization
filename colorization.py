import tensorflow as tf
import numpy as np

learning_rate = 1e-2

# I'm assuming we will resize all images to the same size
# tf.image.resize_images
img_h = 50
img_w = 50

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.zeros(shape))

input = tf.placeholder(tf.float32, [None, img_h, img_w, 1])
output = tf.placeholder(tf.int32, [None, img_h, img_w, 3])

filter_size = 5
conv_size_1 = 32
conv_size_2 = 32
conv_size_3 = 32
hidden_size = 1

# first convolutional layer
W_conv1 = weight_variable([filter_size, filter_size, 1, conv_size_1])
b_conv1 = bias_variable([conv_size_1])
input_reshape = tf.reshape(input, [-1,img_h,img_w,1])
h_conv1 = tf.nn.relu(tf.nn.conv2d(input_reshape, W_conv1, strides=[1,2,2,1], padding='SAME') + b_conv1)

# W_conv2 = weight_variable([filter_size*2, filter_size, 1, conv_size_2])
# b_conv1 = bias_variable([conv_size_1])
# input_reshape = tf.reshape(input, [-1,img_h,img_w,1])
# h_conv1 = tf.nn.relu(tf.nn.conv2d(input_reshape, W_conv1, strides=[1,2,2,1], padding='SAME') + b_conv1)





# # second convolutional layer
W_conv2 = weight_variable([filter_size, filter_size, conv_size_1, conv_size_2])
b_conv2 = bias_variable([conv_size_2])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1,2,2,1], padding='SAME') + b_conv2)


result = tf.reshape(h_conv2, [-1, img_h, img_w, 3, 32])

# # third convolutional layer
# W_conv3 = weight_variable([filter_size, filter_size, conv_size_2, conv_size_3])
# b_conv3 = bias_variable([conv_size_3])
# h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3)

#first feed forward
W_fc1 = weight_variable([img_h/2 * img_w/2 * conv_size_1, hidden_size])
b_fc1 = bias_variable([hidden_size])
h_pool_flat = tf.reshape(h_conv1, [-1, img_h/2 * img_w/2 * conv_size_1])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second feed forward
W_fc2 = weight_variable([hidden_size, img_h * img_w * 3 * 32])
b_fc2 = bias_variable([img_h * img_w * 3 * 32])
result1 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
result = tf.reshape(result1, [-1, img_h, img_w, 3, 32])

loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(result, output))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
        
        sess.run(tf.initialize_all_variables())

#         image = tf.read_file('turkeys/n01794344_17499.JPEG')

        # input_image = tf.image.decode_jpeg(image, channels=1)
        # input_image = tf.image.resize_images(input_image, tf.constant([img_h, img_w]))
        # with open('input.jpeg', 'w') as f:
                # f.write(tf.image.encode_jpeg(tf.cast(input_image, tf.uint8)).eval())
        # input_image = input_image.eval()

        # output_image = tf.image.decode_jpeg(image, channels=3)
        # output_image = tf.image.resize_images(output_image, tf.constant([img_h, img_w]))
        # with open('output.jpeg', 'w') as f:
                # f.write(tf.image.encode_jpeg(tf.cast(output_image, tf.uint8)).eval())
        # output_image = output_image.eval()
        # output_image //= 8

        # feed_dict = {input: [input_image], output: [output_image], keep_prob: .5}

        # for i in range(100):
                # train_loss, _ = sess.run([loss, train_step], feed_dict)
                # print i, train_loss

        # test_result = sess.run(result, feed_dict)
        # test_result = tf.reshape(test_result, [img_h * img_w * 3, 32])
        # test_result = tf.nn.softmax(test_result)
        # test_result = tf.reshape(test_result, [img_h, img_w,  3, 32])
        # test_result = tf.argmax(test_result, 3)
        # test_result *= 8
        # test_result += 4
        # with open('result.jpeg', 'w') as f:
                # f.write(tf.image.encode_jpeg(tf.cast(test_result, tf.uint8)).eval())

        image_nums = []
        for i in range(24628):
                try:
                        open('turkeys/n01794344_{}.JPEG'.format(i))
                        image_nums.append(i)
                except:
                        continue

        train = image_nums[1000:1050]
        test = image_nums[1000:1015]

        print len(train)

        for i in range(5):
            for i in range(len(train)):
                    image = tf.read_file('turkeys/n01794344_{}.JPEG'.format(train[i]))
                    input_image = tf.image.decode_jpeg(image, channels=1)
                    input_image = tf.image.resize_images(input_image, tf.constant([img_h, img_w])).eval()
                    output_image = tf.image.decode_jpeg(image, channels=3)
                    output_image = tf.image.resize_images(output_image, tf.constant([img_h, img_w])).eval()
                    output_image //= 8
                    feed_dict = {input: [input_image], output: [output_image], keep_prob: .5}
                    if i % 10 == 0:
                            train_loss = sess.run(loss, feed_dict)
                            print(i, train_loss)
                    sess.run(train_step, feed_dict)

        print('***************TESTING***************')
        total_loss = 0
        for i in range(len(test)):
                image = tf.read_file('turkeys/n01794344_{}.JPEG'.format(test[i]))
                input_image = tf.image.decode_jpeg(image, channels=1)
                input_image = tf.image.resize_images(input_image, tf.constant([img_h, img_w])).eval()
                output_image = tf.image.decode_jpeg(image, channels=3)
                output_image = tf.image.resize_images(output_image, tf.constant([img_h, img_w])).eval()
                output_image //= 8
                feed_dict = {input: [input_image], output: [output_image], keep_prob: 1}
                test_loss = sess.run(loss, feed_dict)
                total_loss += test_loss
                if i % 10 == 0:
                        print(test[i], test_loss)
                        test_result = sess.run(result, feed_dict)
                        test_result = tf.nn.softmax(tf.reshape(test_result,[img_h * img_w *3,32]))
                        test_result = tf.reshape(test_result, [img_h, img_w, 3, 32])
                        test_result = tf.argmax(test_result, 3)
                        test_result *= 8
                        test_result += 4
                        with open('result{}.jpeg'.format(i), 'w') as f:
                                f.write(tf.image.encode_jpeg(tf.cast(test_result, tf.uint8)).eval())

        avg_loss = total_loss / len(test)
        print('avg_loss {}'.format(avg_loss))


