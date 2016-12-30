from model import Model
import tensorflow as tf

class ModelAllConvMod(Model):
    def __init__(self, activation_func):
        super(ModelAllConvMod, self).__init__("allconvmod", activation_func)
        with tf.device("/gpu:0"):
            self.input = tf.placeholder(tf.float32, shape=[None,32,32,3])
            self.labels = tf.placeholder(tf.float32, shape=[None,10])
            self.dropout = tf.placeholder(tf.float32)

            # 3x3 layer 1
            W_conv1 = self.__weights([3,3,3,96])
            b_conv1 = self.__biases([96])
            h_conv1 = self.act_fn(tf.nn.conv2d(self.input, W_conv1, strides=[1,1,1,1], padding="SAME") + b_conv1)

            # 3x3 layer 2
            W_conv2 = self.__weights([3,3,96,96])
            b_conv2 = self.__biases([96])
            h_conv2 = self.act_fn(tf.nn.conv2d(h_conv1, W_conv2, strides=[1,1,1,1], padding="SAME") + b_conv2)

            # 3x3 downsample layer 3
            W_downsample3 = self.__weights([3,3,96,96])
            b_downsample3 = self.__biases([96])
            h_downsample3 = self.act_fn(tf.nn.conv2d(h_conv2, W_downsample3, strides=[1,2,2,1], padding="SAME") + b_downsample3)

            # 3x3 layer 4
            W_conv4 = self.__weights([3,3,96,192])
            b_conv4 = self.__biases([192])
            h_conv4 = self.act_fn(tf.nn.conv2d(h_downsample3, W_conv4, strides=[1,1,1,1], padding="SAME") + b_conv4)

            # 3x3 layer 5
            W_conv5 = self.__weights([3,3,192,192])
            b_conv5 = self.__biases([192])
            h_conv5 = self.act_fn(tf.nn.conv2d(h_conv4, W_conv5, strides=[1,1,1,1], padding="SAME") + b_conv5)

            # 3x3 downsample layer 6
            W_downsample6 = self.__weights([3,3,192,192])
            b_downsample6 = self.__biases([192])
            h_downsample6 = self.act_fn(tf.nn.conv2d(h_conv5, W_downsample6, strides=[1,2,2,1], padding="SAME") + b_downsample6)

            W_fc1 = self.__weights([8*8*192, 1024])
            b_fc1 = self.__biases([1024])
            h_conv_r3_flat = tf.reshape(h_conv_r3, [-1, 8*8*192])
            h_fc1 = self.act_fn(tf.nn.xw_plus_b(h_conv_r3_flat, W_fc1, b_fc1))

            W_fc2 = self.__weights([1024, 512])
            b_fc2 = self.__biases([512])
            h_fc2 = self.act_fn(tf.nn.xw_plus_b(h_fc1, W_fc2, b_fc2))

            h_fc2_dropout = tf.nn.dropout(h_fc2, self.dropout)

            W_fc3 = self.__weights([512, 10])
            b_fc3 = self.__biases([10])
            self.raw_scores = tf.nn.xw_plus_b(h_fc2_dropout, W_fc3, b_fc3)
            self.probabilities = tf.nn.softmax(self.raw_scores)
            self.prediction = tf.argmax(self.probabilities, 1)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.raw_scores, self.labels))
            self.train = tf.train.AdamOptimizer(1e-4).minimize(loss)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.labels, 1)), tf.float32))

    def train_model(self, session, images, labels):
        session.run(self.train, feed_dict={self.input: images, self.labels: labels, self.dropout: 0.75})

    def predict(self, session, images):
        return session.run(self.prediction, feed_dict={self.input: images, self.dropout: 1.0})

    def get_accuracy(self, session, images, labels):
        return session.run(self.accuracy, feed_dict={self.input: images, self.labels: labels, self.dropout: 1.0})

    def __weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def __biases(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))
