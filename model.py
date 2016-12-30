import tensorflow as tf

class Model:
    def __init__(self, name, activation_func):
        self.name = name
        self.activation_func_name = activation_func
        if activation_func == "relu":
            self.act_fn = tf.nn.relu
        elif activation_func == "elu":
            self.act_fn = tf.nn.elu
        else:
            self.act_fn = tf.nn.relu
            self.activation_func_name = "relu"

    def train(self, session, images, labels):
        raise NotImplementedError()

    def predict(self, session, images):
        raise NotImplementedError()

    def get_accuracy(self, session, images, labels):
        raise NotImplementedError()
