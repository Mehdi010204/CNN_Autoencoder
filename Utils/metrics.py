import numpy as np

def calculate_psnr(original_images, reconstructed_images):
    mse = np.mean(np.square(original_images - reconstructed_images))
    psnr = 10 * np.log10(1 / mse)
    return psnr

def calculate_compression_rate(original_shape, bottleneck_size):
    original_rate = np.prod(original_shape) 
    compressed = bottleneck_size
    compression_rate = original_rate / compressed
    return compression_rate