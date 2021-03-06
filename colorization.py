import tensorflow as tf
import numpy as np

starter_learning_rate = 2e-3
learning_rate_decay = 0.9
decay_steps = 100

# I'm assuming we will resize all images to the same size
# tf.image.resize_images
img_h = 64
img_w = 64

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.zeros(shape))

x = tf.placeholder(tf.float32, [None, img_h, img_w, 1])
output = tf.placeholder(tf.float32, [None, img_h, img_w, 3])

filter_size = 2
conv_size_1 = 64
conv_size_2 = 128
conv_size_3 = 256
conv_size_4 = 256
conv_size_5 = 128
hidden_size = 128

# first convolutional layer
W_conv1 = weight_variable([filter_size, filter_size, 1, conv_size_1])
b_conv1 = bias_variable([conv_size_1])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1,2,2,1], padding='SAME') + b_conv1)

# # second convolutional layer
W_conv2 = weight_variable([filter_size, filter_size, conv_size_1, conv_size_2])
b_conv2 = bias_variable([conv_size_2])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1,2,2,1], padding='SAME') + b_conv2)

# third convolutional layer
W_conv3 = weight_variable([filter_size, filter_size, conv_size_2, conv_size_3])
b_conv3 = bias_variable([conv_size_3])
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3)

# fourth convolutional layer
W_conv4 = weight_variable([filter_size, filter_size, conv_size_3, conv_size_4])
b_conv4 = bias_variable([conv_size_4])
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1,1,1,1], padding='SAME') + b_conv4)

# fifth convolutional layer
W_conv5 = weight_variable([filter_size, filter_size, conv_size_4, conv_size_5])
b_conv5 = bias_variable([conv_size_5])
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1,1,1,1], padding='SAME') + b_conv5)

#first feed forward
W_fc1 = weight_variable([conv_size_5, hidden_size])
b_fc1 = bias_variable([hidden_size])
h_pool_flat = tf.reshape(h_conv5, [-1, conv_size_5])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

#dropou# t
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second feed forward
W_fc2 = weight_variable([hidden_size, img_h * img_w * 3 / 256])
b_fc2 = bias_variable([img_h * img_w * 3 / 256])
result1 = tf.matmul(h_fc1, W_fc2) + b_fc2
result = tf.reshape(result1, [-1, img_h, img_w, 3])

loss = tf.reduce_sum(tf.square(result - output))

global_step = tf.placeholder(tf.int32)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,decay_steps, learning_rate_decay, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)#, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
        
        sess.run(tf.initialize_all_variables())

        # image = tf.read_file('turkeys/n01794344_17499.JPEG')

        # input_image = tf.image.decode_jpeg(image, channels=1)
        # input_image = tf.image.resize_images(input_image, img_h, img_w)
        # with open('input.jpeg', 'w') as f:
        #         f.write(tf.image.encode_jpeg(tf.cast(input_image, tf.uint8)).eval())
        # input_image = input_image.eval()
        # input_image = [input_image]

        # output_image = tf.image.decode_jpeg(image, channels=3)
        # output_image = tf.image.resize_images(output_image, output_h, output_w)
        # with open('output.jpeg', 'w') as f:
        #         f.write(tf.image.encode_jpeg(tf.cast(output_image, tf.uint8)).eval())
        # output_image = output_image.eval()
        # output_image = [output_image]
        # output_image //= 8

        # input_image = [[[[85] for _ in range(256)] for _ in range(256)] for _ in range(10)]
        # output_image = [[[[255, 0, 0] for _ in range(64)] for _ in range(64)] for _ in range(10)]

        # feed_dict = {input: input_image, output: output_image, keep_prob: .5}

        # for i in range(25):
                # train_loss, _ = sess.run([loss, train_step], feed_dict)
                # print i, train_loss

        # test_result = sess.run(result, feed_dict)
        # test_result = tf.reshape(test_result, [img_h * img_w * 3])
        # test_result = tf.nn.softmax(test_result)
        # test_result = tf.reshape(test_result, [img_h, img_w,  3, 32])
        # test_result = tf.argmax(test_result, 3)
        # test_result *= 8
        # test_result += 4
        # with open('result.jpeg', 'w') as f:
        #         f.write(tf.image.encode_jpeg(tf.cast(test_result[0], tf.uint8)).eval())

        image_nums = []
        for i in range(24628):
                try:
                        open('turkeys/n01794344_{}.JPEG'.format(i))
                        image_nums.append(i)
                except:
                        continue

        train = image_nums[:2]
        test = image_nums[:2]

        for batch in range(500):
            input_images = []
            output_images = []
            for i in range(len(train)):
                image = tf.read_file('turkeys/n01794344_{}.JPEG'.format(train[i]))
                input_image = tf.image.decode_jpeg(image, channels=1) 
                input_image = tf.image.resize_images(input_image, tf.constant([img_h, img_w])).eval()
                output_image = tf.image.decode_jpeg(image, channels=3)
                output_image = tf.image.resize_images(output_image, tf.constant([img_h, img_w])).eval()
                input_images.append(input_image)
                output_images.append(output_image)


            feed_dict = {x: input_images, output: output_images, keep_prob: 1,global_step: batch}

            train_loss, _ = sess.run([loss,train_step], feed_dict)
            print batch, train_loss

        save_path = saver.save(sess, "/tmp/model.ckpt")
        print('***************TESTING***************')
        total_loss = 0
        for i in range(len(test)):
            image = tf.read_file('turkeys/n01794344_{}.JPEG'.format(test[i]))
            input_image = tf.image.decode_jpeg(image, channels=1)
            input_image = tf.image.resize_images(input_image, tf.constant([img_h, img_w])).eval()
            output_image = tf.image.decode_jpeg(image, channels=3)
            output_image = tf.image.resize_images(output_image, tf.constant([img_h, img_w])).eval()
            feed_dict = {x: [input_image], output: [output_image], keep_prob: 1,global_step: batch}
            test_loss,test_result = sess.run([loss,result], feed_dict)
            total_loss += test_loss

            # save result
            print(test[i], test_loss)
            test_result = tf.reshape(test_result, [img_h, img_w, 3])
            with open('result{}.jpeg'.format(test[i]), 'w') as f:
                    f.write(tf.image.encode_jpeg(tf.cast(test_result, tf.uint8)).eval())

        avg_loss = total_loss / len(test)
        print('avg_loss {}'.format(avg_loss))


