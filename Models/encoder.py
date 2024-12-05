import tensorflow as tf
from tensorflow.keras import layers

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(16, (5, 5), strides=2, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (5, 5), strides=2, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(256, activation="relu")  # Bottleneck
        ])

    def call(self, inputs):
        return self.encoder(inputs)