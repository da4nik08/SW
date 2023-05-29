import torch
from scipy import signal
import numpy as np
import librosa
import sklearn
from sklearn import preprocessing
from custom_dataset import CustomDataset

def process_data(np_wav, nperseg=1024, samplerate=24000):
    # Function to process a single piece of data
    # Implement your custom processing logic here
    _, _, processed_data = signal.stft(np_wav, samplerate, nperseg=nperseg, return_onesided=True)
    pd = np.power(np.abs(processed_data), 0.5)
    mean = np.mean(pd, axis=1, keepdims=True)
    std = np.std(pd, axis=1, keepdims=True)
    pd = (pd - mean) / std
    new_shape = int(939 * 1024/nperseg)
    pad = np.pad(pd, ((0, 0), (0, new_shape - pd.shape[1])), mode='constant')
    return torch.tensor(np.expand_dims(np.swapaxes(pad,0,1), axis=0))

def process_data_mfcc(np_wav, nperseg=1024, samplerate=24000):
    mfcc = librosa.feature.mfcc(y=np_wav.astype(float), sr=samplerate, hop_length=nperseg)
    pd = sklearn.preprocessing.scale(mfcc, axis=1)
    mean = np.mean(pd, axis=1, keepdims=True)
    std = np.std(pd, axis=1, keepdims=True)
    pd = (pd - mean) / std
    new_shape = int(469 * 1024/nperseg)
    pad = np.pad(pd, ((0, 0), (0, new_shape - pd.shape[1])), mode='constant')
    return torch.tensor(np.expand_dims(np.swapaxes(pad,0,1), axis=0))
    

def collate_fn(batch):
    features_batch, labels_batch = zip(*batch)
    
    processed_batch = []
    for data in features_batch:
        processed_data = process_data_mfcc(data)
        processed_batch.append(processed_data)
    return torch.stack(processed_batch), torch.tensor(labels_batch)