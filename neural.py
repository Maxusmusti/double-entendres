'''
This is the main file for building the neural networks
'''

import numpy as np
import tensorflow as tf
import random

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

        # train
        x_length = len(xs[0])
        self.model.add(tf.keras.layers.Dropout(0.2, input_shape=(x_length,)))
        self.model.add(tf.keras.layers.Dense(200, input_dim=x_length, activation="relu", kernel_initializer='normal'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer='normal'))

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model.fit(xs_training, ys_training, batch_size=20, epochs=10, verbose=1)

    def answer(self, test):
        return self.model(test)
