import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_file_path):
    data = pd.read_csv(csv_file_path).values.astype(np.float32)
    data = data.reshape(-1, 128, 128, 1)
    data /= data.max()  # Normalisation
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data