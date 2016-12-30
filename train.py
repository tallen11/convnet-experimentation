from model_standard import ModelStandard
from model_minmax import ModelMinMax
from model_allconv import ModelAllConv
from model_allconv_mod import ModelAllConvMod
from data_batcher import DataBatcher
from output_writer import OutputWriter
import os
from time import time
import tensorflow as tf

batcher = DataBatcher("cifar")
model = ModelStandard("relu")
# model = ModelMinMax("relu")
# model = ModelAllConv("relu")
# model = ModelAllConvMod("relu")
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
            accuracy = 0
            image_batches, label_batches = batcher.get_test_batches(50)
            for i in range(len(image_batches)):
                accuracy += model.get_accuracy(session, image_batches[i], label_batches[i])
            accuracy /= len(image_batches)

            train_accuracy = 0
            image_batches, label_batches = batcher.get_test_training_batches(50)
            for i in range(len(image_batches)):
                train_accuracy += model.get_accuracy(session, image_batches[i], label_batches[i])
            train_accuracy /= len(image_batches)

            accuracy_data.append(accuracy)
            print("Epoch %i | test acc: %f | train_acc: %f | time: %f" % (epoch_index, accuracy, train_accuracy, time() - epoch_start_time))
            saver.save(session, os.path.join("checkpoints", model.name + ".ckpt"))
            batcher.prepare_epoch()
            step_index = 0
            epoch_start_time = time()
            if epoch_index == epochs:
                break
            epoch_index += 1
        images, labels = batcher.get_batch(batch_size)
        model.train_model(session, images, labels)
        # print("Step %i" % step_index)
        # step_index += 1
print("Training complete")
output = OutputWriter()
output.append_row("results/results.csv", model.name + "-" + model.activation_func_name, accuracy_data)
