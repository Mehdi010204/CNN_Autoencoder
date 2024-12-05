import numpy as np
import matplotlib.pyplot as plt
from Models.autoencoder import Autoencoder
from Utils.data_preprocessing import load_and_preprocess_data
from Utils.visualization import visualize_reconstructions
import Utils.metrics as metrics

# Load the data
train_data, test_data = load_and_preprocess_data("data.csv")

# Compiling the model
decoded_imgs = Autoencoder.predict(test_data)

# Visualiazing the reconstructed images
visualize_reconstructions(test_data, decoded_imgs, num_images=5, random_selection=True)

# PSNR
psnr = metrics.calculate_psnr(test_data, decoded_imgs)
print(f"PSNR: {psnr:.2f} dB")

# Compression rate
compression_rate = metrics.calculate_compression_rate(test_data, decoded_imgs)
print(f"Compression rate: {compression_rate:.2f}")