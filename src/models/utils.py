import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def load_audio(file_path, sr=16000, duration=1.0):
    """Load and preprocess audio file"""
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    
    # Pad or trim to fixed length
    target_length = int(sr * duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    return audio

def create_spectrogram(audio, n_fft=2048, hop_length=512):
    """Convert audio to spectrogram"""
    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(spectrogram)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return spectrogram

def normalize_spectrogram(spectrogram):
    """Normalize spectrogram values to [0,1] range"""
    spec_min = spectrogram.min()
    spec_max = spectrogram.max()
    return (spectrogram - spec_min) / (spec_max - spec_min)

def plot_spectrogram(spectrogram, title='Spectrogram'):
    """Plot a spectrogram"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    return plt

def get_files_list(data_dir, split='train'):
    """Get list of audio files and labels"""
    data_path = Path(data_dir) / split
    files = []
    labels = []
    
    for digit_folder in sorted(data_path.glob('[0-9]')):
        digit = int(digit_folder.name)
        for audio_file in digit_folder.glob('*.wav'):
            files.append(str(audio_file))
            labels.append(digit)
            
    return files, labels

def prepare_batch(batch):
    """Convert batch of spectrograms to tensor"""
    return torch.FloatTensor(np.stack(batch))
