import pytest
import torch
import numpy as np
from src.models.digit_rnn import DigitRNN

@pytest.fixture
def model():
    return DigitRNN()

@pytest.fixture
def mock_input():
    # Create mock spectrogram batch (batch_size, channels, time, freq)
    return torch.randn(4, 1, 128, 128)

def test_model_initialization(model):
    assert isinstance(model, DigitRNN)
    assert hasattr(model, 'rnn')
    assert hasattr(model, 'fc')

def test_model_forward(model, mock_input):
    output = model(mock_input)
    # Should output probabilities for 10 digits (0-9)
    assert output.shape == (4, 10)
    assert torch.allclose(output.sum(dim=1), torch.ones(4))

def test_model_output_range(model, mock_input):
    output = model(mock_input)
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)

def test_model_parameters(model):
    params = list(model.parameters())
    assert len(params) > 0
    assert all(isinstance(p, torch.Tensor) for p in params)

def test_model_training_mode(model):
    model.train()
    assert model.training
    model.eval()
    assert not model.training

def test_model_device_transfer(model):
    if torch.cuda.is_available():
        model = model.cuda()
        assert next(model.parameters()).is_cuda

def test_model_backward_pass(model, mock_input):
    output = model(mock_input)
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 10, (4,)))
    loss.backward()
    # Check if gradients were computed
    assert all(p.grad is not None for p in model.parameters())

if __name__ == '__main__':
    pytest.main([__file__])