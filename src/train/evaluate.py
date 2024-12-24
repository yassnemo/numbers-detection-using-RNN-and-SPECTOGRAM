import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from ..models.digit_rnn import DigitRNN
from ..utils import load_audio, create_spectrogram, get_files_list

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load trained model"""
    model = DigitRNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_single(model, audio_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Predict single audio file"""
    audio = load_audio(audio_path)
    spec = create_spectrogram(audio)
    spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(spec)
        pred = output.argmax(dim=1).item()
    return pred

def evaluate_model(model, data_dir, batch_size=32, split='test'):
    """Evaluate model on test set"""
    device = next(model.parameters()).device
    files, labels = get_files_list(data_dir, split)
    
    predictions = []
    true_labels = []
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        
        # Process batch
        specs = []
        for file in batch_files:
            audio = load_audio(file)
            spec = create_spectrogram(audio)
            specs.append(spec)
        
        # Make predictions
        specs_tensor = torch.FloatTensor(np.stack(specs)).unsqueeze(1).to(device)
        with torch.no_grad():
            outputs = model(specs_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()
        
        predictions.extend(preds)
        true_labels.extend(batch_labels)
    
    return np.array(predictions), np.array(true_labels)

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    model_path = Path('models/digit_rnn.pth')
    data_dir = Path('data')
    
    # Load model
    model = load_model(model_path)
    
    # Evaluate
    predictions, true_labels = evaluate_model(model, data_dir)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Plot results
    plot_confusion_matrix(true_labels, predictions)

if __name__ == '__main__':
    main()