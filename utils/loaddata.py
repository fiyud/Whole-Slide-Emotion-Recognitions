import os
import numpy as np
import librosa
from torch.utils.data import Dataset
import sys

# sys.path.append("F:/2025/Whole-Slide-Emotion-Recognitions/utils")

from conversation import wave_mfcc

def load_data_dual(data_folder):
    waves = []
    mfccs = []
    labels = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".wav"):
                link = os.path.join(root, file)
                wave, mfcc, label = wave_mfcc(link)

                waves.extend(wave)
                mfccs.extend(mfcc)
                labels.extend(label)

    return np.array(waves), np.array(mfccs), np.array(labels)

class EmoDualDataset(Dataset):
    def __init__(self, waves, mfccs, labels):
        self.waves = waves
        self.mfccs = mfccs
        self.labels = labels

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, index):
        return self.waves[index], self.mfccs[index], self.labels[index]