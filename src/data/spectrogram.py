import librosa
import numpy as np
import torch

class SpectrogramGenerator:
    def __init__(self, sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def generate(self, audio_path):
        # Load audio file
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_norm = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
        
        return torch.FloatTensor(mel_spec_norm)