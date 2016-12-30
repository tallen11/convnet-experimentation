from model_standard import ModelStandard
from model_minmax import ModelMinMax
from model_allconv import ModelAllConv
from data_batcher import DataBatcher
from output_writer import OutputWriter
import os
from time import time
import tensorflow as tf

batcher = DataBatcher("cifar")
# model = ModelStandard("relu")
# model = ModelMinMax("relu")
model = ModelAllConv("relu")
saver = tf.train.Saver()

epochs = 100
batch_size = 500
with tf.Session() as session:
    print("Beginning training...")
    session.run(tf.global_variables_initializer())
    epoch_index = 0
    accuracy_data = []
    # step_index = 0
    epoch_start_time = time()
    while True:
        if batcher.epoch_finished():
            images, labels = batcher.get_test_batch()
            accuracy = model.get_accuracy(session, images, labels)
            accuracy_data.append(accuracy)
            print("Epoch %i ~ %f ~ %f" % (epoch_index, accuracy, time() - epoch_start_time))
            saver.save(session, os.path.join("checkpoints", model.name + ".ckpt"))
            batcher.prepare_epoch()
            epoch_index += 1
            step_index = 0
            epoch_start_time = time()
            if epoch_index == epochs:
                break
        images, labels = batcher.get_batch(batch_size)
        model.train_model(session, images, labels)
        # print("Step %i" % step_index)
        # step_index += 1
print("Training complete")
output = OutputWriter()
output.append_row("results/results.csv", model.name + "-" + model.activation_func_name, accuracy_data)
