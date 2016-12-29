from model import Model
import tensorflow as tf

class ModelStandard(Model):
    def __init__(self):
        super(ModelStandard, self).__init__("standard")
        with tf.device("/gpu:0"):
            self.input = tf.placeholder(tf.float32, shape=[None,32,32,3])
            self.labels = tf.placeholder(tf.float32, shape=[None,10])
            self.dropout = tf.placeholder(tf.float32)

            W_conv1 = self.__weights([5,5,3,32])
            b_conv1 = self.__biases([32])
            h_conv1 = tf.nn.elu(tf.nn.conv2d(self.input, W_conv1, strides=[1,1,1,1], padding="SAME") + b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

            W_conv2 = self.__weights([5,5,32,64])
            b_conv2 = self.__biases([64])
            h_conv2 = tf.nn.elu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding="SAME") + b_conv2)
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

            W_fc1 = self.__weights([8 * 8 * 64, 1024])
            b_fc1 = self.__biases([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
            h_fc1 = tf.nn.elu(tf.nn.xw_plus_b(h_pool2_flat, W_fc1, b_fc1))

            h_fc1_dropout = tf.nn.dropout(h_fc1, self.dropout)

            W_fc2 = self.__weights([1024, 10])
            b_fc2 = self.__biases([10])

            self.raw_scores = tf.nn.xw_plus_b(h_fc1_dropout, W_fc2, b_fc2)
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
