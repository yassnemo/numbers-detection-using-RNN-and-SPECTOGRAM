import os
import torch
from torch.utils.data import Dataset
from .spectrogram import SpectrogramGenerator

class DigitDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.spec_gen = SpectrogramGenerator()
        
        # Load all audio files and their labels
        for digit in range(10):
            digit_dir = os.path.join(data_dir, str(digit))
            for audio_file in os.listdir(digit_dir):
                if audio_file.endswith('.wav'):
                    self.samples.append({
                        'path': os.path.join(digit_dir, audio_file),
                        'label': digit
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        spectrogram = self.spec_gen.generate(sample['path'])
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
            
        return spectrogram, label