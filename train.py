import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from Models.autoencoder import Autoencoder
from Utils.data_preprocessing import load_and_preprocess_data
from Utils.custom_loss import hybrid_loss

# Charger les données
train_data, test_data = load_and_preprocess_data("data.csv")

# Initialiser et compiler le modèle
autoencoder = Autoencoder()
autoencoder.compile(optimizer="adam", loss=hybrid_loss)
autoencoder.summary()

# Early stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

# Training the model
autoencoder.fit(
    train_data, train_data,
    epochs=30,
    batch_size=64,
    shuffle=True,
    validation_data=(test_data, test_data),
    callbacks=[early_stopping]
)
