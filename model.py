class Model:
    def __init__(self, name):
        self.name = name

    def train(self, session, images, labels):
        raise NotImplementedError()

    def predict(self, session, images):
        raise NotImplementedError()

    def get_accuracy(self, session, images, labels):
        raise NotImplementedError()
