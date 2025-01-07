import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from Models.autoencoder import Autoencoder
from Utils.data_preprocessing import load_and_preprocess_data
from Utils.custom_loss import hybrid_loss
from Utils import metrics, visualization

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

# Compiling the model
decoded_imgs = Autoencoder.predict(test_data)

# Visualiazing the reconstructed images
visualization.visualize_reconstructions(test_data, decoded_imgs, num_images=5, random_selection=True)

# PSNR
psnr = metrics.calculate_psnr(test_data, decoded_imgs)
print(f"PSNR: {psnr:.2f} dB")

# Compression rate
compression_rate = metrics.calculate_compression_rate(test_data, decoded_imgs)
print(f"Compression rate: {compression_rate:.2f}")