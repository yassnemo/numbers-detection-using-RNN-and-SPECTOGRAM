import pytest
import numpy as np
import torch
import os
from pathlib import Path
from src.utils import load_audio, create_spectrogram, get_files_list, prepare_batch

@pytest.fixture
def sample_audio_path(tmp_path):
    # Create a test WAV file
    import scipy.io.wavfile as wav
    sample_rate = 16000
    duration = 1
    t = np.linspace(0, duration, sample_rate)
    audio = np.sin(2*np.pi*440*t)  # 440 Hz sine wave
    wav_path = tmp_path / "test.wav"
    wav.write(wav_path, sample_rate, audio.astype(np.float32))
    return wav_path

@pytest.fixture
def mock_dataset_structure(tmp_path):
    # Create mock dataset structure
    for split in ['train', 'val']:
        for digit in range(10):
            path = tmp_path / split / str(digit)
            path.mkdir(parents=True)
            wav_path = path / "sample.wav"
            # Create empty wav file
            wav_path.touch()
    return tmp_path

def test_load_audio(sample_audio_path):
    audio = load_audio(sample_audio_path)
    assert isinstance(audio, np.ndarray)
    assert len(audio) == 16000  # 1 second at 16kHz
    assert not np.any(np.isnan(audio))

def test_create_spectrogram():
    audio = np.random.randn(16000)
    spec = create_spectrogram(audio)
    assert isinstance(spec, np.ndarray)
    assert len(spec.shape) == 2
    assert np.all((spec >= 0) & (spec <= 1))

def test_get_files_list(mock_dataset_structure):
    files, labels = get_files_list(mock_dataset_structure, split='train')
    assert len(files) == 10  # One file per digit
    assert len(labels) == 10
    assert all(isinstance(l, int) for l in labels)
    assert all(0 <= l <= 9 for l in labels)

def test_prepare_batch():
    batch = [np.random.randn(128, 128) for _ in range(4)]
    tensor = prepare_batch(batch)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (4, 128, 128)
    assert tensor.dtype == torch.float32

def test_load_audio_invalid_file():
    with pytest.raises(FileNotFoundError):
        load_audio("nonexistent.wav")

def test_create_spectrogram_empty_audio():
    with pytest.raises(ValueError):
        create_spectrogram(np.array([]))

def test_get_files_list_empty_directory(tmp_path):
    files, labels = get_files_list(tmp_path, split='train')
    assert len(files) == 0
    assert len(labels) == 0

def test_prepare_batch_empty_list():
    with pytest.raises(ValueError):
        prepare_batch([])

if __name__ == '__main__':
    pytest.main([__file__])