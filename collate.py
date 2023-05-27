import torch
from scipy import signal
import numpy as np

def process_data(np_wav, nperseg=1024, samplerate=24000):
    # Function to process a single piece of data
    # Implement your custom processing logic here
    _, _, processed_data = signal.stft(np_wav, samplerate, nperseg=nperseg)  # Example processing: doubling the data
    pd = np.power(np.abs(processed_data), 0.5)
    mean = np.mean(pd, axis=1, keepdims=True)
    std = np.std(pd, axis=1, keepdims=True)
    pd = (pd - mean) / std
    new_shape = int(939 * 1024/nperseg)
    return torch.tensor(np.pad(pd, ((0, 0), (0, new_shape - pd.shape[1])), mode='constant'))

def collate_fn(batch):
    features_batch, labels_batch = zip(*batch)
    
    processed_batch = []
    for data in features_batch:
        processed_data = process_data(data)
        processed_batch.append(processed_data)
    return torch.stack(processed_batch), torch.stack(labels_batch)