import os
import numpy as np
import librosa

n_mfcc = 128
window_size = 2048
strides = 512
window_size_stft = 1024
window = np.hanning(window_size_stft)

def load_emodata_dual(link, sr=16000, duration=5):
    wave_form, _ = librosa.load(path=link, sr=sr)
    labels = os.path.basename(os.path.dirname(link))

    if len(wave_form) < sr * duration:
        wave_form = np.pad(wave_form, (0, sr * duration - len(wave_form)), 'symmetric')
        mfcc1 = librosa.feature.mfcc(y=wave_form, sr=8000, n_mfcc=n_mfcc, n_fft=window_size, hop_length=strides)
        return np.array([wave_form]), np.array([mfcc1]), np.array([labels])

    elif len(wave_form) == sr * duration:
        mfcc2 = librosa.feature.mfcc(y=wave_form, sr=8000, n_mfcc=n_mfcc, n_fft=window_size, hop_length=strides)
        return np.array([wave_form]), np.array([mfcc2]), np.array([labels])

    else:
        wave_segments = []
        _labels = [labels] * (len(wave_form) // (sr * duration) + 1)
        for i in range(0, len(wave_form), sr * duration):
            wave_segments.append(wave_form[i:i + sr * duration])

        len_wave_segments_last = len(wave_segments[-1])
        padding = sr * duration - len_wave_segments_last
        temp = np.append(wave_segments[-2][sr * duration - padding:], wave_segments[-1])
        wave_segments[-1] = temp

        mfcc_seg = librosa.feature.mfcc(y=np.array(wave_segments), sr=8000, n_mfcc=n_mfcc, n_fft=window_size, hop_length=strides)
        
        return np.array(wave_segments), np.array(mfcc_seg), np.array(_labels)