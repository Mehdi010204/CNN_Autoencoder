import matplotlib.pyplot as plt
import numpy as np

def visualize_reconstructions(original_images, reconstructed_images, num_images=5, random_selection=True):
    if random_selection:
        # Select random indices from the dataset
        indices = np.random.choice(len(original_images), num_images, replace=False)
    else:
        # Use the first `num_images` indices if random selection is disabled
        indices = range(num_images)
    
    # Create the plot
    plt.figure(figsize=(20, 8))
    for i, idx in enumerate(indices):
        # Display original image
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[idx].reshape(128, 128), cmap="gray")
        plt.title(f"Original {i + 1}")
        plt.axis("off")  # Hide axes

        # Display reconstructed image
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[idx].reshape(128, 128), cmap="gray")
        plt.title(f"Reconstructed {i + 1}")
        plt.axis("off")  # Hide axes

    plt.tight_layout()
    plt.show()
