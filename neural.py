'''
This is the main file for building the neural networks
'''

import numpy as np
import tensorflow as tf

class Model:

    def __init__(self):
        self.model = tf.keras.Sequential()

    def train(self, xs, ys):
        nb_training = int(0.9 * len(xs))
        xs_training = np.array(xs[:nb_training])
        ys_training = np.array(ys[:nb_training])
        xs_validation = np.array(xs[nb_training:])
        ys_validation = np.array(ys[nb_training:])

        print(f"{len(xs_training)} training examples")
        print(f"{len(xs_validation)} validation examples")

        x_length = len(xs[0])
        self.model.add(tf.keras.layers.Dropout(0.2, input_shape=(x_length,)))
        self.model.add(tf.keras.layers.Dense(200, input_dim=x_length, activation="relu", kernel_initializer='normal'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer='normal'))

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model.fit(xs_training, ys_training, batch_size=100, epochs=10, verbose=1)
