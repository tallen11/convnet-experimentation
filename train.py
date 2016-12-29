from model_standard import ModelStandard
from model_minmax import ModelMinMax
from data_batcher import DataBatcher
import os
import tensorflow as tf

batcher = DataBatcher("cifar")
model = ModelStandard()
# model_mm = ModelMinMax()
saver = tf.train.Saver()

epochs = 1000
batch_size = 20
with tf.Session() as session:
    epoch_index = 0
    while True:
        if batcher.epoch_finished():
            images, labels = batcher.get_test_batch()
            # run thru model
            print("Epoch %i ~ $f" % (epoch_index, 0.1))
            # saver.save(session, os.path.join("checkpoints", model.get_name()))
            batcher.prepare_epoch()
            epoch_index += 1
            if epoch_index == epochs:
                print("Training complete")
                break
        images, labels = batcher.get_batch(batch_size)
        print(images, labels)
