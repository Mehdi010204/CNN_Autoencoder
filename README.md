# CNN Autoencoder for Image Compression

This repository contains a Convolutional Neural Network (CNN) based Autoencoder for image compression and reconstruction. The autoencoder model is implemented using TensorFlow/Keras and is designed to work with grayscale images of size 128x128 pixels.

The CNN Autoencoder in this project learns to compress and reconstruct images. It uses a combination of Convolutional layers, Batch Normalization, and Transposed Convolution layers to learn image representations. The model aims to minimize both the Mean Squared Error (MSE) loss and the Structural Similarity Index (SSIM) for high-quality reconstructions.

The project is structured in multiple Python scripts that allow for easy training, evaluation, and visualization of the results. The model is designed to work with grayscale images (128x128 pixels) and compresses them into a smaller representation while maintaining reconstruction quality.

## Requirements
The following libraries are required to run the code:
- `tensorflow` >= 2.x
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install the required dependencies using `pip`:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
