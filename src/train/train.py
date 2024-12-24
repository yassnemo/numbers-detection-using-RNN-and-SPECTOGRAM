import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..data.preprocessing import DigitDataset
from ..models.rnn_model import DigitRNN

def train(data_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
    # Create datasets
    train_dataset = DigitDataset(data_dir + '/train')
    val_dataset = DigitDataset(data_dir + '/val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = DigitRNN(
        input_size=128,  # Matches n_mels from SpectrogramGenerator
        hidden_size=256,
        num_layers=2,
        num_classes=10
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(num_epochs):
        model.train()
        for batch_specs, batch_labels in train_loader:
            batch_specs, batch_labels = batch_specs.to(device), batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_specs)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_specs, batch_labels in val_loader:
                batch_specs, batch_labels = batch_specs.to(device), batch_labels.to(device)
                outputs = model(batch_specs)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
        print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train('/f/data/dataset')