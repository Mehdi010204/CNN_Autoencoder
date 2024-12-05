import tensorflow as tf
from tensorflow.keras import layers

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = tf.keras.Sequential([
            layers.Dense(16 * 16 * 32, activation="relu"),
            layers.Reshape((16, 16, 32)),
            layers.Conv2DTranspose(128, (5, 5), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(64, (5, 5), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(32, (5, 5), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")
        ])

    def call(self, encoded):
        return self.decoder(encoded)