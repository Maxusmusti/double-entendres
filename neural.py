'''
This is the main file for building the neural networks
'''

import numpy as np
import tensorflow as tf
import random
from keras import callbacks

class Model:

    def __init__(self):
        self.model = tf.keras.Sequential()

    def train(self, xs, ys):
        # randomly shuffle data
        pair = []
        for i in range(len(xs)):
            sen = xs[i]
            label = ys[i]
            pair.append([sen, label])

        random.shuffle(pair)
        xs = []
        ys = []
        for item in pair:
            sen = item[0]
            label = item[1]
            xs.append(sen)
            ys.append(label)

        # split training and testing data
        nb_training = int(0.8 * len(xs))
        xs_training = np.array(xs[:nb_training])
        ys_training = np.array(ys[:nb_training])
        xs_validation = np.array(xs[nb_training:])
        ys_validation = np.array(ys[nb_training:])

        print(f"{len(xs_training)} training examples")
        print(f"{len(xs_validation)} validation examples")

        earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                                mode="min", patience=5,
                                                restore_best_weights=True)

        # train
        x_length = len(xs[0])
        self.model.add(tf.keras.layers.Dropout(0.2, input_shape=(x_length,)))
        self.model.add(tf.keras.layers.Dense(200, input_dim=x_length, activation="relu", kernel_initializer='normal'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(10))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer='normal'))

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        # Early stopping to learn epoh limit
        train_data = self.model.fit(np.array(xs), np.array(ys), batch_size=64, epochs=25, verbose=1, validation_split=0.2, callbacks =[earlystopping])

        # No epoch limit
        ###train_data = self.model.fit(np.array(xs), np.array(ys), batch_size=64, epochs=12, verbose=1, validation_split=0.2)

        return train_data

    def predict(self, xs):
        return self.model.predict(xs)
