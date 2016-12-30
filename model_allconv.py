from model import Model
import tensorflow as tf

class ModelAllConv(Model):
    def __init__(self, activation_func):
        super(ModelAllConv, self).__init__("allconv", activation_func)
        with tf.device("/gpu:0"):
            self.input = tf.placeholder(tf.float32, shape=[None,32,32,3])
            self.labels = tf.placeholder(tf.float32, shape=[None,10])
            self.dropout = tf.placeholder(tf.float32)

            W_conv1 = self.__weights([5,5,3,32])
            b_conv1 = self.__biases([32])
            h_conv1 = self.act_fn(tf.nn.conv2d(self.input, W_conv1, strides=[1,1,1,1], padding="SAME") + b_conv1)
            W_conv_r1 = self.__weights([5,5,32,32])
            b_conv_r1 = self.__biases([32])
            h_conv_r1 = tf.nn.conv2d(h_conv1, W_conv_r1, strides=[1,2,2,1], padding="SAME") + b_conv_r1

            W_conv2 = self.__weights([5,5,32,32])
            b_conv2 = self.__biases([32])
            h_conv2 = self.act_fn(tf.nn.conv2d(h_conv_r1, W_conv2, strides=[1,1,1,1], padding="SAME") + b_conv2)
            W_conv_r2 = self.__weights([5,5,32,32])
            b_conv_r2 = self.__biases([32])
            h_conv_r2 = tf.nn.conv2d(h_conv2, W_conv_r2, strides=[1,2,2,1], padding="SAME") + b_conv_r2

            W_conv3 = self.__weights([5,5,32,64])
            b_conv3 = self.__biases([64])
            h_conv3 = self.act_fn(tf.nn.conv2d(h_conv_r2, W_conv3, strides=[1,1,1,1], padding="SAME") + b_conv3)
            W_conv_r3 = self.__weights([5,5,64,64])
            b_conv_r3 = self.__biases([64])
            h_conv_r3 = tf.nn.conv2d(h_conv3, W_conv_r3, strides=[1,2,2,1], padding="SAME") + b_conv_r3

            W_fc1 = self.__weights([4*4*64, 1024])
            b_fc1 = self.__biases([1024])
            h_conv_r3_flat = tf.reshape(h_conv_r3, [-1, 4*4*64])
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
