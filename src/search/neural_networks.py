import os
from multiprocessing.dummy import active_children

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_text as tft


class AutoEncoder(tf.keras.Model):
    def __init__(self, vocab_size: int, bottleneck_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden1 = tf.keras.layers.Dense(units=512, activation="swish")
        # self.hidden2 = tf.keras.layers.Dense(units=512, activation="swish")
        # self.hidden3 = tf.keras.layers.Dense(units=512, activation="swish")
        # self.hidden4 = tf.keras.layers.Dense(units=512, activation="swish")
        # self.hidden5 = tf.keras.layers.Dense(units=512, activation="swish")
        self.bottleneck = tf.keras.layers.Dense(units=bottleneck_dim, activation="relu")
        self.out = tf.keras.layers.Dense(units=vocab_size, activation="relu")

    def call(self, inputs, return_vector: bool = False):
        x = self.hidden1(inputs)
        # x = self.hidden2(x)
        # x = self.hidden3(x)
        # x = self.hidden4(x)
        # x = self.hidden5(x)
        x = self.bottleneck(x)

        if return_vector:
            return x

        x = self.out(x)
        return x
